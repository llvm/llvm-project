# Exception classes used when testing the Python pretty printers.
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

class NoLineError(Exception):
    """Custom exception to indicate that a test file doesn't contain
    the requested string.
    """

    def __init__(self, file_name, string):
        """Constructor.

        Args:
            file_name (string): The name of the test file.
            string (string): The string that was requested.
        """

        super(NoLineError, self).__init__()
        self.file_name = file_name
        self.string = string

    def __str__(self):
        """Shows a readable representation of the exception."""

        return ('File {0} has no line containing the following string: {1}'
                .format(self.file_name, self.string))

class DebugError(Exception):
    """Custom exception to indicate that a required debugging symbol is missing.
    """

    def __init__(self, symbol):
        """Constructor.

        Args:
            symbol (string): The name of the entity whose debug info is missing.
        """

        super(DebugError, self).__init__()
        self.symbol = symbol

    def __str__(self):
        """Shows a readable representation of the exception."""

        return ('The required debugging information for {0} is missing.'
                .format(self.symbol))
