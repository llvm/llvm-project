#!/usr/bin/python3
# Consistency checks for glibc system call lists.
# Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
# <http://www.gnu.org/licenses/>.

import argparse
import sys

import glibcextract
import glibcsyscalls

def main():
    """The main entry point."""
    parser = argparse.ArgumentParser(
        description="System call list consistency checks")
    parser.add_argument('--cc', metavar='CC', required=True,
                        help='C compiler (including options) to use')
    parser.add_argument('syscall_numbers_list', metavar='PATH',
                        help='Path to the list of system call numbers')
    parser.add_argument('syscall_names_list', metavar='PATH',
                        help='Path to the list of system call names')

    args = parser.parse_args()

    glibc_constants = glibcsyscalls.load_arch_syscall_header(
        args.syscall_numbers_list)
    with open(args.syscall_names_list) as inp:
        glibc_names = glibcsyscalls.SyscallNamesList(inp)
    kernel_constants = glibcsyscalls.kernel_constants(args.cc)
    kernel_version = glibcsyscalls.linux_kernel_version(args.cc)

    errors = 0
    warnings = False
    for name in glibc_constants.keys() & kernel_constants.keys():
        if glibc_constants[name] != kernel_constants[name]:
            print("error: syscall {!r} number mismatch: glibc={!r} kernel={!r}"
                  .format(name, glibc_constants[name], kernel_constants[name]))
            errors = 1

    # The architecture-specific list in the glibc tree must be a
    # subset of the global list of system call names.
    for name in glibc_constants.keys() - set(glibc_names.syscalls):
        print("error: architecture syscall {!r} missing from global names list"
              .format(name))
        errors = 1

    for name in glibc_constants.keys() - kernel_constants.keys():
        print("info: glibc syscall {!r} not known to kernel".format(name))
        warnings = True

    # If the glibc-recorded kernel version is not older than the
    # installed kernel headers, the glibc system call set must be a
    # superset of the kernel system call set.
    if glibc_names.kernel_version >= kernel_version:
        for name in kernel_constants.keys() - glibc_constants.keys():
            print("error: kernel syscall {!r} ({}) not known to glibc"
                  .format(name, kernel_constants[name]))
            errors = 1
    else:
        for name in kernel_constants.keys() - glibc_constants.keys():
            print("warning: kernel syscall {!r} ({}) not known to glibc"
                  .format(name, kernel_constants[name]))
            warnings = True

    if errors > 0 or warnings:
        print("info: glibc tables are based on kernel version {}".format(
            ".".join(map(str, glibc_names.kernel_version))))
        print("info: installed kernel headers are version {}".format(
            ".".join(map(str, kernel_version))))

    sys.exit(errors)

if __name__ == '__main__':
    main()
