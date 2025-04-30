# Tests for the MutexPrinter class.
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

import sys

from test_printers_common import *

test_source = sys.argv[1]
test_bin = sys.argv[2]
printer_files = sys.argv[3:]
printer_names = ['global glibc-pthread-locks']

try:
    init_test(test_bin, printer_files, printer_names)
    go_to_main()

    var = 'mutex'
    to_string = 'pthread_mutex_t'

    break_at(test_source, 'Test status (destroyed)')
    continue_cmd() # Go to test_status_destroyed
    test_printer(var, to_string, {'Status': 'Destroyed'})

    break_at(test_source, 'Test status (non-robust)')
    continue_cmd() # Go to test_status_no_robust
    test_printer(var, to_string, {'Status': 'Not acquired'})
    next_cmd()
    thread_id = get_current_thread_lwpid()
    # Owner ID might be reported either as the thread ID or as "Unknown"
    # (if e.g. lock elision is enabled).
    test_printer(var, to_string,
                 {'Status': 'Acquired, possibly with no waiters',
                  'Owner ID': r'({0}|Unknown)'.format(thread_id)})

    break_at(test_source, 'Test status (robust)')
    continue_cmd() # Go to test_status_robust
    test_printer(var, to_string, {'Status': 'Not acquired'})

    # We'll now test the robust mutex locking states.  We'll create a new
    # thread that will lock a robust mutex and exit without unlocking it.
    break_at(test_source, 'Create')
    continue_cmd() # Go to test_locking_state_robust
    # Set a breakpoint for the new thread to hit.
    break_at(test_source, 'Thread function')
    continue_cmd()
    # By now the new thread is created and has hit its breakpoint.
    set_scheduler_locking(True)
    parent = 1
    child = 2
    select_thread(child)
    child_id = get_current_thread_lwpid()
    # We've got the new thread's ID.
    select_thread(parent)
    # Make the new thread finish its function while we wait.
    continue_cmd(thread=child)
    # The new thread should be dead by now.
    break_at(test_source, 'Test locking (robust)')
    continue_cmd()
    test_printer(var, to_string, {'Owner ID': r'{0} \(dead\)'.format(child_id)})
    # Try to lock and unlock the mutex.
    next_cmd()
    test_printer(var, to_string, {'Owner ID': thread_id,
                           'State protected by this mutex': 'Inconsistent'})
    next_cmd()
    test_printer(var, to_string, {'Status': 'Not acquired',
                        'State protected by this mutex': 'Not recoverable'})
    set_scheduler_locking(False)

    break_at(test_source, 'Test recursive locks')
    continue_cmd() # Go to test_recursive_locks
    test_printer(var, to_string, {'Times acquired by the owner': '2'})
    next_cmd()
    test_printer(var, to_string, {'Times acquired by the owner': '3'})
    continue_cmd() # Exit

except (NoLineError, pexpect.TIMEOUT) as exception:
    print('Error: {0}'.format(exception))
    result = FAIL

else:
    print('Test succeeded.')
    result = PASS

exit(result)
