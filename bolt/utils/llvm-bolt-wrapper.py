#!/usr/bin/env python3
import argparse
import subprocess
from typing import *
import tempfile
import copy
import os
import shutil
import sys
import re
import configparser
from types import SimpleNamespace
from textwrap import dedent

# USAGE:
# 0. Prepare two BOLT build versions: base and compare.
# 1. Create the config by invoking this script with required options.
#    Save the config as `llvm-bolt-wrapper.ini` next to the script or
#    in the testing directory.
# In the base BOLT build directory:
# 2. Rename `llvm-bolt` to `llvm-bolt.real`
# 3. Create a symlink from this script to `llvm-bolt`
# 4. Create `llvm-bolt-wrapper.ini` and fill it using the example below.
#
# This script will compare binaries produced by base and compare BOLT, and
# report elapsed processing time and max RSS.

# read options from config file llvm-bolt-wrapper.ini in script CWD
#
# [config]
# # mandatory
# base_bolt = /full/path/to/llvm-bolt.real
# cmp_bolt = /full/path/to/other/llvm-bolt
# # optional, default to False
# verbose
# keep_tmp
# no_minimize
# run_sequentially
# compare_output
# skip_binary_cmp
# # optional, defaults to timing.log in CWD
# timing_file = timing1.log

def read_cfg():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = configparser.ConfigParser(allow_no_value = True)
    cfgs = cfg.read("llvm-bolt-wrapper.ini")
    if not cfgs:
        cfgs = cfg.read(os.path.join(src_dir, "llvm-bolt-wrapper.ini"))
    assert cfgs, f"llvm-bolt-wrapper.ini is not found in {os.getcwd()}"

    def get_cfg(key):
        # if key is not present in config, assume False
        if key not in cfg['config']:
            return False
        # if key is present, but has no value, assume True
        if not cfg['config'][key]:
            return True
        # if key has associated value, interpret the value
        return cfg['config'].getboolean(key)

    d = {
        # BOLT binary locations
        'BASE_BOLT': cfg['config']['base_bolt'],
        'CMP_BOLT': cfg['config']['cmp_bolt'],
        # optional
        'VERBOSE': get_cfg('verbose'),
        'KEEP_TMP': get_cfg('keep_tmp'),
        'NO_MINIMIZE': get_cfg('no_minimize'),
        'RUN_SEQUENTIALLY': get_cfg('run_sequentially'),
        'COMPARE_OUTPUT': get_cfg('compare_output'),
        'SKIP_BINARY_CMP': get_cfg('skip_binary_cmp'),
        'TIMING_FILE': cfg['config'].get('timing_file', 'timing.log'),
    }
    if d['VERBOSE']:
        print(f"Using config {os.path.abspath(cfgs[0])}")
    return SimpleNamespace(**d)

# perf2bolt mode
PERF2BOLT_MODE = ['-aggregate-only', '-ignore-build-id']

# boltdiff mode
BOLTDIFF_MODE = ['-diff-only', '-o', '/dev/null']

# options to suppress binary differences as much as possible
MINIMIZE_DIFFS = ['-bolt-info=0']

# bolt output options that need to be intercepted
BOLT_OUTPUT_OPTS = {
    '-o': 'BOLT output binary',
    '-w': 'BOLT recorded profile',
}

# regex patterns to exclude the line from log comparison
SKIP_MATCH = [
    'BOLT-INFO: BOLT version',
    r'^Args: ',
    r'^BOLT-DEBUG:',
    r'BOLT-INFO:.*data.*output data',
    'WARNING: reading perf data directly',
]

def run_cmd(cmd, out_f, cfg):
    if cfg.VERBOSE:
        print(' '.join(cmd))
    return subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.STDOUT)

def run_bolt(bolt_path, bolt_args, out_f, cfg):
    p2b = os.path.basename(sys.argv[0]) == 'perf2bolt' # perf2bolt mode
    bd = os.path.basename(sys.argv[0]) == 'llvm-boltdiff' # boltdiff mode
    hm = sys.argv[1] == 'heatmap' # heatmap mode
    cmd = ['/usr/bin/time', '-f', '%e %M', bolt_path] + bolt_args
    if p2b:
        # -ignore-build-id can occur at most once, hence remove it from cmd
        if '-ignore-build-id' in cmd:
            cmd.remove('-ignore-build-id')
        cmd += PERF2BOLT_MODE
    elif bd:
        cmd += BOLTDIFF_MODE
    elif not cfg.NO_MINIMIZE and not hm:
        cmd += MINIMIZE_DIFFS
    return run_cmd(cmd, out_f, cfg)

def prepend_dash(args: Mapping[AnyStr, AnyStr]) -> Sequence[AnyStr]:
    '''
    Accepts parsed arguments and returns flat list with dash prepended to
    the option.
    Example: Namespace(o='test.tmp') -> ['-o', 'test.tmp']
    '''
    dashed = [('-'+key,value) for (key,value) in args.items()]
    flattened = list(sum(dashed, ()))
    return flattened

def replace_cmp_path(tmp: AnyStr, args: Mapping[AnyStr, AnyStr]) -> Sequence[AnyStr]:
    '''
    Keeps file names, but replaces the path to a temp folder.
    Example: Namespace(o='abc/test.tmp') -> Namespace(o='/tmp/tmpf9un/test.tmp')
    Except preserve /dev/null.
    '''
    replace_path = lambda x: os.path.join(tmp, os.path.basename(x)) if x != '/dev/null' else '/dev/null'
    new_args = {key: replace_path(value) for key, value in args.items()}
    return prepend_dash(new_args)

def preprocess_args(args: argparse.Namespace) -> Mapping[AnyStr, AnyStr]:
    '''
    Drop options that weren't parsed (e.g. -w), convert to a dict
    '''
    return {key: value for key, value in vars(args).items() if value}

def write_to(txt, filename, mode='w'):
    with open(filename, mode) as f:
        f.write(txt)

def wait(proc, fdesc):
    proc.wait()
    fdesc.close()
    return open(fdesc.name)

def compare_logs(main, cmp, skip_begin=0, skip_end=0, str_input=True):
    '''
    Compares logs but allows for certain lines to be excluded from comparison.
    If str_input is True (default), the input it assumed to be a string,
    which is split into lines. Otherwise the input is assumed to be a file.
    Returns None on success, mismatch otherwise.
    '''
    main_inp = main.splitlines() if str_input else main.readlines()
    cmp_inp = cmp.splitlines() if str_input else cmp.readlines()
    # rewind logs after consumption
    if not str_input:
        main.seek(0)
        cmp.seek(0)
    for lhs, rhs in list(zip(main_inp, cmp_inp))[skip_begin:-skip_end or None]:
        if lhs != rhs:
            # check skip patterns
            for skip in SKIP_MATCH:
                # both lines must contain the pattern
                if re.search(skip, lhs) and re.search(skip, rhs):
                    break
            # otherwise return mismatching lines
            else:
                return (lhs, rhs)
    return None

def fmt_cmp(cmp_tuple):
    if not cmp_tuple:
        return ''
    return f'main:\n{cmp_tuple[0]}\ncmp:\n{cmp_tuple[1]}\n'

def compare_with(lhs, rhs, cmd, skip_begin=0, skip_end=0):
    '''
    Runs cmd on both lhs and rhs and compares stdout.
    Returns tuple (mismatch, lhs_stdout):
        - if stdout matches between two files, mismatch is None,
        - otherwise mismatch is a tuple of mismatching lines.
    '''
    run = lambda binary: subprocess.run(cmd.split() + [binary],
                                        text=True, check=True,
                                        capture_output=True).stdout
    run_lhs = run(lhs)
    run_rhs = run(rhs)
    cmp = compare_logs(run_lhs, run_rhs, skip_begin, skip_end)
    return cmp, run_lhs

def parse_cmp_offset(cmp_out):
    '''
    Extracts byte number from cmp output:
    file1 file2 differ: byte X, line Y
    '''
    return int(re.search(r'byte (\d+),', cmp_out).groups()[0])

def report_real_time(binary, main_err, cmp_err, cfg):
    '''
    Extracts real time from stderr and appends it to TIMING FILE it as csv:
    "output binary; base bolt; cmp bolt"
    '''
    def get_real_from_stderr(logline):
        return '; '.join(logline.split())
    for line in main_err:
        pass
    main = get_real_from_stderr(line)
    for line in cmp_err:
        pass
    cmp = get_real_from_stderr(line)
    write_to(f"{binary}; {main}; {cmp}\n", cfg.TIMING_FILE, 'a')
    # rewind logs after consumption
    main_err.seek(0)
    cmp_err.seek(0)

def clean_exit(tmp, out, exitcode, cfg):
    # temp files are only cleaned on success
    if not cfg.KEEP_TMP:
        shutil.rmtree(tmp)

    # report stdout and stderr from the main process
    shutil.copyfileobj(out, sys.stdout)
    sys.exit(exitcode)

def find_section(offset, readelf_hdr):
    hdr = readelf_hdr.split('\n')
    section = None
    # extract sections table (parse objdump -hw output)
    for line in hdr[5:-1]:
        cols = line.strip().split()
        # extract section offset
        file_offset = int(cols[5], 16)
        # section size
        size = int(cols[2], 16)
        if offset >= file_offset and offset <= file_offset + size:
            if sys.stdout.isatty(): # terminal supports colors
                print(f"\033[1m{line}\033[0m")
            else:
                print(f">{line}")
            section = cols[1]
        else:
            print(line)
    return section

def main_config_generator():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_bolt', help='Full path to base llvm-bolt binary')
    parser.add_argument('cmp_bolt', help='Full path to cmp llvm-bolt binary')
    parser.add_argument('--verbose', action='store_true',
                        help='Print subprocess invocation cmdline (default False)')
    parser.add_argument('--keep_tmp', action='store_true',
                        help = 'Preserve tmp folder on a clean exit '
                        '(tmp directory is preserved on crash by default)')
    parser.add_argument('--no_minimize', action='store_true',
                        help=f'Do not add `{MINIMIZE_DIFFS}` that is used '
                        'by default to reduce binary differences')
    parser.add_argument('--run_sequentially', action='store_true',
                        help='Run both binaries sequentially (default '
                        'in parallel). Use for timing comparison')
    parser.add_argument('--compare_output', action='store_true',
                        help = 'Compare bolt stdout/stderr (disabled by default)')
    parser.add_argument('--skip_binary_cmp', action='store_true',
                        help = 'Disable output comparison')
    parser.add_argument('--timing_file', help = 'Override path to timing log '
                        'file (default `timing.log` in CWD)')
    args = parser.parse_args()

    print(dedent(f'''\
    [config]
    # mandatory
    base_bolt = {args.base_bolt}
    cmp_bolt = {args.cmp_bolt}'''))
    del args.base_bolt
    del args.cmp_bolt
    d = vars(args)
    if any(d.values()):
        print("# optional")
        for key, value in d.items():
            if value:
                print(key)

def main():
    cfg = read_cfg()
    # intercept output arguments
    parser = argparse.ArgumentParser(add_help=False)
    for option, help in BOLT_OUTPUT_OPTS.items():
        parser.add_argument(option, help=help)
    args, unknownargs = parser.parse_known_args()
    args = preprocess_args(args)
    cmp_args = copy.deepcopy(args)
    tmp = tempfile.mkdtemp()
    cmp_args = replace_cmp_path(tmp, cmp_args)

    # reconstruct output arguments: prepend dash
    args = prepend_dash(args)

    # run both BOLT binaries
    main_f = open(os.path.join(tmp, 'main_bolt.stdout'), 'w')
    cmp_f = open(os.path.join(tmp, 'cmp_bolt.stdout'), 'w')
    main_bolt = run_bolt(cfg.BASE_BOLT, unknownargs + args, main_f, cfg)
    if cfg.RUN_SEQUENTIALLY:
        main_out = wait(main_bolt, main_f)
        cmp_bolt = run_bolt(cfg.CMP_BOLT, unknownargs + cmp_args, cmp_f, cfg)
    else:
        cmp_bolt = run_bolt(cfg.CMP_BOLT, unknownargs + cmp_args, cmp_f, cfg)
        main_out = wait(main_bolt, main_f)
    cmp_out = wait(cmp_bolt, cmp_f)

    # check exit code
    if main_bolt.returncode != cmp_bolt.returncode:
        print(tmp)
        exit("exitcode mismatch")

    # don't compare output upon unsuccessful exit
    if main_bolt.returncode != 0:
        cfg.SKIP_BINARY_CMP = True

    # compare logs, skip_end=1 skips the line with time
    out = compare_logs(main_out, cmp_out, skip_end=1, str_input=False) if cfg.COMPARE_OUTPUT else None
    if out:
        print(tmp)
        print(fmt_cmp(out))
        write_to(fmt_cmp(out), os.path.join(tmp, 'summary.txt'))
        exit("logs mismatch")

    if os.path.basename(sys.argv[0]) == 'llvm-boltdiff': # boltdiff mode
        # no output binary to compare, so just exit
        clean_exit(tmp, main_out, main_bolt.returncode, cfg)

    # compare binaries (using cmp)
    main_binary = args[args.index('-o')+1]
    cmp_binary = cmp_args[cmp_args.index('-o')+1]
    if main_binary == '/dev/null':
        assert cmp_binary == '/dev/null'
        cfg.SKIP_BINARY_CMP = True

    # report binary timing as csv: output binary; base bolt real; cmp bolt real
    report_real_time(main_binary, main_out, cmp_out, cfg)

    if not cfg.SKIP_BINARY_CMP:
        # check if files exist
        main_exists = os.path.exists(main_binary)
        cmp_exists = os.path.exists(cmp_binary)
        if main_exists and cmp_exists:
            # proceed to comparison
            pass
        elif not main_exists and not cmp_exists:
            # both don't exist, assume it's intended, skip comparison
            clean_exit(tmp, main_out, main_bolt.returncode, cfg)
        elif main_exists:
            assert not cmp_exists
            exit(f"{cmp_binary} doesn't exist")
        else:
            assert not main_exists
            exit(f"{main_binary} doesn't exist")

        cmp_proc = subprocess.run(['cmp', '-b', main_binary, cmp_binary],
                                  capture_output=True, text=True)
        if cmp_proc.returncode:
            # check if output is an ELF file (magic bytes)
            with open(main_binary, 'rb') as f:
                magic = f.read(4)
                if magic != b'\x7fELF':
                    exit("output mismatch")
            # check if ELF headers match
            mismatch, _ = compare_with(main_binary, cmp_binary, 'readelf -We')
            if mismatch:
                print(fmt_cmp(mismatch))
                write_to(fmt_cmp(mismatch), os.path.join(tmp, 'headers.txt'))
                exit("headers mismatch")
            # if headers match, compare sections (skip line with filename)
            mismatch, hdr = compare_with(main_binary, cmp_binary, 'objdump -hw',
                                         skip_begin=2)
            assert not mismatch
            # check which section has the first mismatch
            mismatch_offset = parse_cmp_offset(cmp_proc.stdout)
            section = find_section(mismatch_offset, hdr)
            exit(f"binary mismatch @{hex(mismatch_offset)} ({section})")

    clean_exit(tmp, main_out, main_bolt.returncode, cfg)

if __name__ == "__main__":
    # config generator mode if the script is launched as is
    if os.path.basename(__file__) == "llvm-bolt-wrapper.py":
        main_config_generator()
    else:
        # llvm-bolt interceptor mode otherwise
        main()
