# Common tests for the ConditionVariablePrinter class.
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

    var = 'condvar'
    to_string = 'pthread_cond_t'

    break_at(test_source, 'Test status (destroyed)')
    continue_cmd() # Go to test_status_destroyed
    test_printer(var, to_string, {'Threads known to still execute a wait function': '0'})

    continue_cmd() # Exit

except (NoLineError, pexpect.TIMEOUT) as exception:
    print('Error: {0}'.format(exception))
    result = FAIL

else:
    print('Test succeeded.')
    result = PASS

exit(result)
