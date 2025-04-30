#!/usr/bin/python3
# Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
"""Benchmark output validator

Given a benchmark output file in json format and a benchmark schema file,
validate the output against the schema.
"""

from __future__ import print_function
import json
import sys
import os

try:
    import import_bench as bench
except ImportError:
    print('Import Error: Output will not be validated.')
    # Return success because we don't want the bench target to fail just
    # because the jsonschema module was not found.
    sys.exit(os.EX_OK)


def print_and_exit(message, exitcode):
    """Prints message to stderr and returns the exit code.

    Args:
        message: The message to print
        exitcode: The exit code to return

    Returns:
        The passed exit code
    """
    print(message, file=sys.stderr)
    return exitcode


def main(args):
    """Main entry point

    Args:
        args: The command line arguments to the program

    Returns:
        0 on success or a non-zero failure code

    Exceptions:
        Exceptions thrown by validate_bench
    """
    if len(args) != 2:
        return print_and_exit("Usage: %s <bench.out file> <bench.out schema>"
                % sys.argv[0], os.EX_USAGE)

    try:
        bench.parse_bench(args[0], args[1])
    except IOError as e:
        return print_and_exit("IOError(%d): %s" % (e.errno, e.strerror),
                os.EX_OSFILE)

    except bench.validator.ValidationError as e:
        return print_and_exit("Invalid benchmark output: %s" % e.message,
            os.EX_DATAERR)

    except bench.validator.SchemaError as e:
        return print_and_exit("Invalid schema: %s" % e.message, os.EX_DATAERR)

    print("Benchmark output in %s is valid." % args[0])
    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
