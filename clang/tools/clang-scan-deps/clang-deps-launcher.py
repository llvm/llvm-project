#!/usr/bin/env python3
"""
Launcher for clang which computes the dependency for the file being compiled.
"""
import sys
import os
import subprocess
import time
import json
import tempfile
import fcntl


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))


def trim_clang_args_in_cdb(cdb):
    """Trim -Xclang options away since they are not portable"""
    # We also replace the clang exe path so it can find the clang headers.
    clang_exe = os.path.basename(cdb['arguments'][0])
    trim_args = [os.path.join(TOOLS_DIR, clang_exe)]
    skip = False
    for arg in cdb['arguments'][1:]:
        if skip:
            skip = False
            continue
        if arg == '-Xclang':
            skip = True
            continue
        trim_args.append(arg)
    cdb['arguments'] = trim_args


def main():
    # Clang command is the input.
    clang_cmd = sys.argv[1:]
    # Figure out the output path.
    output_idx = len(clang_cmd) - clang_cmd[::-1].index('-o')
    output_object = clang_cmd[output_idx]
    # Write compilation data base to STDOUT, and strip the last ',' so it is a
    # valid JSON object.
    clang_cmd.extend(['-MJ', '-'])
    start = time.perf_counter()
    cdb = subprocess.check_output(clang_cmd).strip()[:-1]
    end = time.perf_counter()
    output_size = os.path.getsize(output_object)

    # load and clean up compiler commands, write to a temp file.
    cdb_object = json.loads(cdb)
    trim_clang_args_in_cdb(cdb_object)

    cdb_output = [ cdb_object ]
    cdb_file = tempfile.NamedTemporaryFile(suffix='.cdb', mode='w+')
    json.dump(cdb_output, cdb_file)
    cdb_file.flush()

    # run clang-scan-deps to collect dependency.
    cas_path = os.getenv('CLANG_SCAN_DEPS_CAS_PATH')
    if cas_path is None:
        scan_deps_extra_opt = '--in-memory-cas'
    else:
        scan_deps_extra_opt = '--cas-path=' + cas_path
    scan_deps = os.path.join(TOOLS_DIR, "clang-scan-deps")
    scan_deps_cmd = [scan_deps, '-compilation-database', cdb_file.name,
                     '-format', 'experimental-tree-full', scan_deps_extra_opt]

    scan_deps_output = subprocess.check_output(scan_deps_cmd)
    cdb_file.close()

    # Write output.
    scan_deps_object = json.loads(scan_deps_output)
    scan_deps_object['time'] = end - start
    scan_deps_object['size'] = output_size

    # Write outoput file.
    # See if the output file environmental variable is set, if set, use
    # accumulated json file format. If not, write next to output object.
    accumulated_file = os.getenv('CLANG_SCAN_DEPS_OUTPUT_FILE')
    if accumulated_file is None:
        output_filename = output_object + '.stats'
        with open(output_filename, 'w+') as of:
            json.dump(scan_deps_object, of)
        return

    # Write accumulated JSON object file format
    json_str = json.dumps(scan_deps_object)
    with open(accumulated_file, 'a+') as of:
        fcntl.flock(of, fcntl.LOCK_EX)  # lock file for write
        of.write(str(len(json_str)))    # write the size of the JSON blob.
        of.write('\n')                  # new line
        of.write(json_str)              # write json blob
        of.write('\n')                  # new line
        fcntl.flock(of, fcntl.LOCK_UN)  # unlock output file


if __name__ == '__main__':
  main()
