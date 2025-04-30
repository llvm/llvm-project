# Common tests for the MutexPrinter and MutexAttributesPrinter classes.
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
PRIOCEILING = 42

try:
    init_test(test_bin, printer_files, printer_names)
    go_to_main()

    check_debug_symbol('struct pthread_mutexattr')

    mutex_var = 'mutex'
    mutex_to_string = 'pthread_mutex_t'

    attr_var = 'attr'
    attr_to_string = 'pthread_mutexattr_t'

    break_at(test_source, 'Set type')
    continue_cmd() # Go to test_settype
    next_cmd(2)
    test_printer(attr_var, attr_to_string, {'Type': 'Error check'})
    test_printer(mutex_var, mutex_to_string, {'Type': 'Error check'})
    next_cmd(2)
    test_printer(attr_var, attr_to_string, {'Type': 'Recursive'})
    test_printer(mutex_var, mutex_to_string, {'Type': 'Recursive'})
    next_cmd(2)
    test_printer(attr_var, attr_to_string, {'Type': 'Normal'})
    test_printer(mutex_var, mutex_to_string, {'Type': 'Normal'})

    break_at(test_source, 'Set robust')
    continue_cmd() # Go to test_setrobust
    next_cmd(2)
    test_printer(attr_var, attr_to_string, {'Robust': 'Yes'})
    test_printer(mutex_var, mutex_to_string, {'Robust': 'Yes'})
    next_cmd(2)
    test_printer(attr_var, attr_to_string, {'Robust': 'No'})
    test_printer(mutex_var, mutex_to_string, {'Robust': 'No'})

    break_at(test_source, 'Set shared')
    continue_cmd() # Go to test_setpshared
    next_cmd(2)
    test_printer(attr_var, attr_to_string, {'Shared': 'Yes'})
    test_printer(mutex_var, mutex_to_string, {'Shared': 'Yes'})
    next_cmd(2)
    test_printer(attr_var, attr_to_string, {'Shared': 'No'})
    test_printer(mutex_var, mutex_to_string, {'Shared': 'No'})

    break_at(test_source, 'Set protocol')
    continue_cmd() # Go to test_setprotocol
    next_cmd(2)
    test_printer(attr_var, attr_to_string, {'Protocol': 'Priority inherit'})
    test_printer(mutex_var, mutex_to_string, {'Protocol': 'Priority inherit'})
    next_cmd(2)
    test_printer(attr_var, attr_to_string, {'Protocol': 'Priority protect'})
    test_printer(mutex_var, mutex_to_string, {'Protocol': 'Priority protect'})
    next_cmd(2)
    test_printer(mutex_var, mutex_to_string, {'Priority ceiling':
                                              str(PRIOCEILING)})
    next_cmd()
    test_printer(attr_var, attr_to_string, {'Protocol': 'None'})
    test_printer(mutex_var, mutex_to_string, {'Protocol': 'None'})

    continue_cmd() # Exit

except (NoLineError, pexpect.TIMEOUT) as exception:
    print('Error: {0}'.format(exception))
    result = FAIL

except DebugError as exception:
    print(exception)
    result = UNSUPPORTED

else:
    print('Test succeeded.')
    result = PASS

exit(result)
