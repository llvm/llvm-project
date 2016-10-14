#===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

import ast
import distutils.spawn
import signal
import subprocess
import sys


def execute_command(cmd, input_str=None):
    """
    Execute a command, capture and return its output.
    """
    kwargs = {
        'stdin': subprocess.PIPE,
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
    }
    p = subprocess.Popen(cmd, **kwargs)
    out, err = p.communicate(input=input_str)
    exitCode = p.wait()
    if exitCode == -signal.SIGINT:
        raise KeyboardInterrupt
    return out, err, exitCode


def execute_command_verbose(cmd, input_str=None):
    """
    Execute a command and print its output on failure.
    """
    out, err, exitCode = execute_command(cmd, input_str=input_str)
    if exitCode != 0:
        report = "Command: %s\n" % ' '.join(["'%s'" % a for a in cmd])
        report += "Exit Code: %d\n" % exitCode
        if out:
            report += "Standard Output:\n--\n%s--" % out
        if err:
            report += "Standard Error:\n--\n%s--" % err
        report += "\n\nFailed!"
        sys.stderr.write('%s\n' % report)
    return out, err, exitCode


def read_syms_from_list(slist):
    """
    Read a list of symbols from a list of strings.
    Each string is one symbol.
    """
    return [ast.literal_eval(l) for l in slist]


def read_syms_from_file(filename):
    """
    Read a list of symbols in from a file.
    """
    with open(filename, 'r') as f:
        data = f.read()
    return read_syms_from_list(data.splitlines())


def read_blacklist(filename):
    with open(filename, 'r') as f:
        data = f.read()
    lines = [l.strip() for l in data.splitlines() if l.strip()]
    lines = [l for l in lines if not l.startswith('#')]
    return lines


def write_syms(sym_list, out=None, names_only=False):
    """
    Write a list of symbols to the file named by out.
    """
    out_str = ''
    out_list = sym_list
    out_list.sort(key=lambda x: x['name'])
    if names_only:
        out_list = [sym['name'] for sym in sym_list]
    for sym in out_list:
        out_str += '%s\n' % sym
    if out is None:
        sys.stdout.write(out_str)
    else:
        with open(out, 'w') as f:
            f.write(out_str)


_cppfilt_exe = distutils.spawn.find_executable('c++filt')


def demangle_symbol(symbol):
    if _cppfilt_exe is None:
        return symbol
    out, _, exit_code = execute_command_verbose(
        [_cppfilt_exe], input_str=symbol)
    if exit_code != 0:
        return symbol
    return out


def is_elf(filename):
    with open(filename, 'r') as f:
        magic_bytes = f.read(4)
    return magic_bytes == '\x7fELF'


def is_mach_o(filename):
    with open(filename, 'r') as f:
        magic_bytes = f.read(4)
    return magic_bytes in [
        '\xfe\xed\xfa\xce',  # MH_MAGIC
        '\xce\xfa\xed\xfe',  # MH_CIGAM
        '\xfe\xed\xfa\xcf',  # MH_MAGIC_64
        '\xcf\xfa\xed\xfe',  # MH_CIGAM_64
        '\xca\xfe\xba\xbe',  # FAT_MAGIC
        '\xbe\xba\xfe\xca'   # FAT_CIGAM
    ]


def is_library_file(filename):
    if sys.platform == 'darwin':
        return is_mach_o(filename)
    else:
        return is_elf(filename)


def extract_or_load(filename):
    import sym_check.extract
    if is_library_file(filename):
        return sym_check.extract.extract_symbols(filename)
    return read_syms_from_file(filename)
