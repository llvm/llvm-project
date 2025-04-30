# Common functions and variables for testing the Python pretty printers.
#
# Copyright (C) 2016-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
#
# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

"""These tests require PExpect 4.0 or newer.

Exported constants:
    PASS, FAIL, UNSUPPORTED (int): Test exit codes, as per evaluate-test.sh.
"""

import os
import re
from test_printers_exceptions import *

PASS = 0
FAIL = 1
UNSUPPORTED = 77

gdb_bin = 'gdb'
gdb_options = '-q -nx'
gdb_invocation = '{0} {1}'.format(gdb_bin, gdb_options)
pexpect_min_version = 4
gdb_min_version = (7, 8)
encoding = 'utf-8'

try:
    import pexpect
except ImportError:
    print('PExpect 4.0 or newer must be installed to test the pretty printers.')
    exit(UNSUPPORTED)

pexpect_version = pexpect.__version__.split('.')[0]

if int(pexpect_version) < pexpect_min_version:
    print('PExpect 4.0 or newer must be installed to test the pretty printers.')
    exit(UNSUPPORTED)

if not pexpect.which(gdb_bin):
    print('gdb 7.8 or newer must be installed to test the pretty printers.')
    exit(UNSUPPORTED)

timeout = 5
TIMEOUTFACTOR = os.environ.get('TIMEOUTFACTOR')

if TIMEOUTFACTOR:
    timeout = int(TIMEOUTFACTOR)

# Otherwise GDB is run in interactive mode and readline may send escape
# sequences confusing output for pexpect.
os.environ["TERM"]="dumb"

try:
    # Check the gdb version.
    version_cmd = '{0} --version'.format(gdb_invocation, timeout=timeout)
    gdb_version_out = pexpect.run(version_cmd, encoding=encoding)

    # The gdb version string is "GNU gdb <PKGVERSION><version>", where
    # PKGVERSION can be any text.  We assume that there'll always be a space
    # between PKGVERSION and the version number for the sake of the regexp.
    version_match = re.search(r'GNU gdb .* ([1-9][0-9]*)\.([0-9]+)',
                              gdb_version_out)

    if not version_match:
        print('The gdb version string (gdb -v) is incorrectly formatted.')
        exit(UNSUPPORTED)

    gdb_version = (int(version_match.group(1)), int(version_match.group(2)))

    if gdb_version < gdb_min_version:
        print('gdb 7.8 or newer must be installed to test the pretty printers.')
        exit(UNSUPPORTED)

    # Check if gdb supports Python.
    gdb_python_cmd = '{0} -ex "python import os" -batch'.format(gdb_invocation,
                                                                timeout=timeout)
    gdb_python_error = pexpect.run(gdb_python_cmd, encoding=encoding)

    if gdb_python_error:
        print('gdb must have python support to test the pretty printers.')
        print('gdb output: {!r}'.format(gdb_python_error))
        exit(UNSUPPORTED)

    # If everything's ok, spawn the gdb process we'll use for testing.
    gdb = pexpect.spawn(gdb_invocation, echo=False, timeout=timeout,
                        encoding=encoding)
    gdb_prompt = u'\(gdb\)'
    gdb.expect(gdb_prompt)

except pexpect.ExceptionPexpect as exception:
    print('Error: {0}'.format(exception))
    exit(FAIL)

def test(command, pattern=None):
    """Sends 'command' to gdb and expects the given 'pattern'.

    If 'pattern' is None, simply consumes everything up to and including
    the gdb prompt.

    Args:
        command (string): The command we'll send to gdb.
        pattern (raw string): A pattern the gdb output should match.

    Returns:
        string: The string that matched 'pattern', or an empty string if
            'pattern' was None.
    """

    match = ''

    gdb.sendline(command)

    if pattern:
        # PExpect does a non-greedy match for '+' and '*'.  Since it can't look
        # ahead on the gdb output stream, if 'pattern' ends with a '+' or a '*'
        # we may end up matching only part of the required output.
        # To avoid this, we'll consume 'pattern' and anything that follows it
        # up to and including the gdb prompt, then extract 'pattern' later.
        index = gdb.expect([u'{0}.+{1}'.format(pattern, gdb_prompt),
                            pexpect.TIMEOUT])

        if index == 0:
            # gdb.after now contains the whole match.  Extract the text that
            # matches 'pattern'.
            match = re.match(pattern, gdb.after, re.DOTALL).group()
        elif index == 1:
            # We got a timeout exception.  Print information on what caused it
            # and bail out.
            error = ('Response does not match the expected pattern.\n'
                     'Command: {0}\n'
                     'Expected pattern: {1}\n'
                     'Response: {2}'.format(command, pattern, gdb.before))

            raise pexpect.TIMEOUT(error)
    else:
        # Consume just the the gdb prompt.
        gdb.expect(gdb_prompt)

    return match

def init_test(test_bin, printer_files, printer_names):
    """Loads the test binary file and the required pretty printers to gdb.

    Args:
        test_bin (string): The name of the test binary file.
        pretty_printers (list of strings): A list with the names of the pretty
            printer files.
    """

    # Load all the pretty printer files.  We're assuming these are safe.
    for printer_file in printer_files:
        test('source {0}'.format(printer_file))

    # Disable all the pretty printers.
    test('disable pretty-printer', r'0 of [0-9]+ printers enabled')

    # Enable only the required printers.
    for printer in printer_names:
        test('enable pretty-printer {0}'.format(printer),
             r'[1-9][0-9]* of [1-9]+ printers enabled')

    # Finally, load the test binary.
    test('file {0}'.format(test_bin))

    # Disable lock elision.
    test('set environment GLIBC_TUNABLES glibc.elision.enable=0')

def go_to_main():
    """Executes a gdb 'start' command, which takes us to main."""

    test('start', r'main')

def get_line_number(file_name, string):
    """Returns the number of the line in which 'string' appears within a file.

    Args:
        file_name (string): The name of the file we'll search through.
        string (string): The string we'll look for.

    Returns:
        int: The number of the line in which 'string' appears, starting from 1.
    """
    number = -1

    with open(file_name) as src_file:
        for i, line in enumerate(src_file):
            if string in line:
                number = i + 1
                break

    if number == -1:
        raise NoLineError(file_name, string)

    return number

def break_at(file_name, string, temporary=True, thread=None):
    """Places a breakpoint on the first line in 'file_name' containing 'string'.

    'string' is usually a comment like "Stop here".  Notice this may fail unless
    the comment is placed inline next to actual code, e.g.:

        ...
        /* Stop here */
        ...

    may fail, while:

        ...
        some_func(); /* Stop here */
        ...

    will succeed.

    If 'thread' isn't None, the breakpoint will be set for all the threads.
    Otherwise, it'll be set only for 'thread'.

    Args:
        file_name (string): The name of the file we'll place the breakpoint in.
        string (string): A string we'll look for inside the file.
            We'll place a breakpoint on the line which contains it.
        temporary (bool): Whether the breakpoint should be automatically deleted
            after we reach it.
        thread (int): The number of the thread we'll place the breakpoint for,
            as seen by gdb.  If specified, it should be greater than zero.
    """

    if not thread:
        thread_str = ''
    else:
        thread_str = 'thread {0}'.format(thread)

    if temporary:
        command = 'tbreak'
        break_type = 'Temporary breakpoint'
    else:
        command = 'break'
        break_type = 'Breakpoint'

    line_number = str(get_line_number(file_name, string))

    test('{0} {1}:{2} {3}'.format(command, file_name, line_number, thread_str),
         r'{0} [0-9]+ at 0x[a-f0-9]+: file {1}, line {2}\.'.format(break_type,
                                                                   file_name,
                                                                   line_number))

def continue_cmd(thread=None):
    """Executes a gdb 'continue' command.

    If 'thread' isn't None, the command will be applied to all the threads.
    Otherwise, it'll be applied only to 'thread'.

    Args:
        thread (int): The number of the thread we'll apply the command to,
            as seen by gdb.  If specified, it should be greater than zero.
    """

    if not thread:
        command = 'continue'
    else:
        command = 'thread apply {0} continue'.format(thread)

    test(command)

def next_cmd(count=1, thread=None):
    """Executes a gdb 'next' command.

    If 'thread' isn't None, the command will be applied to all the threads.
    Otherwise, it'll be applied only to 'thread'.

    Args:
        count (int): The 'count' argument of the 'next' command.
        thread (int): The number of the thread we'll apply the command to,
            as seen by gdb.  If specified, it should be greater than zero.
    """

    if not thread:
        command = 'next'
    else:
        command = 'thread apply {0} next'

    test('{0} {1}'.format(command, count))

def select_thread(thread):
    """Selects the thread indicated by 'thread'.

    Args:
        thread (int): The number of the thread we'll switch to, as seen by gdb.
            This should be greater than zero.
    """

    if thread > 0:
        test('thread {0}'.format(thread))

def get_current_thread_lwpid():
    """Gets the current thread's Lightweight Process ID.

    Returns:
        string: The current thread's LWP ID.
    """

    # It's easier to get the LWP ID through the Python API than the gdb CLI.
    command = 'python print(gdb.selected_thread().ptid[1])'

    return test(command, r'[0-9]+')

def set_scheduler_locking(mode):
    """Executes the gdb 'set scheduler-locking' command.

    Args:
        mode (bool): Whether the scheduler locking mode should be 'on'.
    """
    modes = {
        True: 'on',
        False: 'off'
    }

    test('set scheduler-locking {0}'.format(modes[mode]))

def test_printer(var, to_string, children=None, is_ptr=True):
    """ Tests the output of a pretty printer.

    For a variable called 'var', this tests whether its associated printer
    outputs the expected 'to_string' and children (if any).

    Args:
        var (string): The name of the variable we'll print.
        to_string (raw string): The expected output of the printer's 'to_string'
            method.
        children (map {raw string->raw string}): A map with the expected output
            of the printer's children' method.
        is_ptr (bool): Whether 'var' is a pointer, and thus should be
            dereferenced.
    """

    if is_ptr:
        var = '*{0}'.format(var)

    test('print {0}'.format(var), to_string)

    if children:
        for name, value in children.items():
            # Children are shown as 'name = value'.
            test('print {0}'.format(var), r'{0} = {1}'.format(name, value))

def check_debug_symbol(symbol):
    """ Tests whether a given debugging symbol exists.

    If the symbol doesn't exist, raises a DebugError.

    Args:
        symbol (string): The symbol we're going to check for.
    """

    try:
        test('ptype {0}'.format(symbol), r'type = {0}'.format(symbol))

    except pexpect.TIMEOUT:
        # The symbol doesn't exist.
        raise DebugError(symbol)
