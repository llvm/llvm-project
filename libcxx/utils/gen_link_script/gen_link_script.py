#!/usr/bin/env python
#===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

import os
import sys

def print_and_exit(msg):
    sys.stderr.write(msg + '\n')
    sys.exit(1)

def usage_and_exit():
    print_and_exit("Usage: ./gen_link_script.py [--help] [--dryrun] <path/to/libcxx.so> <abi_libname>")

def help_and_exit():
    help_msg = \
"""Usage

  gen_link_script.py [--help] [--dryrun] <path/to/libcxx.so> <abi_libname>

  Generate a linker script that links libc++ to the proper ABI library.
  The script replaces the specified libc++ symlink.
  An example script for c++abi would look like "INPUT(libc++.so.1 -lc++abi)".

Arguments
  <path/to/libcxx.so> - The top level symlink to the versioned libc++ shared
                        library. This file is replaced with a linker script.
  <abi_libname>       - The name of the ABI library to use in the linker script.
                        The name must be one of [c++abi, stdc++, supc++, cxxrt].

Exit Status:
  0 if OK,
  1 if the action failed.
"""
    print_and_exit(help_msg)

def parse_args():
    args = list(sys.argv)
    del args[0]
    if len(args) == 0:
        usage_and_exit()
    if args[0] == '--help':
        help_and_exit()
    dryrun = '--dryrun' == args[0]
    if dryrun:
        del args[0]
    if len(args) != 2:
        usage_and_exit()
    symlink_file = args[0]
    abi_libname = args[1]
    return dryrun, symlink_file, abi_libname

def main():
    dryrun, symlink_file, abi_libname = parse_args()

    # Check that the given libc++.so file is a valid symlink.
    if not os.path.islink(symlink_file):
        print_and_exit("symlink file %s is not a symlink" % symlink_file)

    # Read the symlink so we know what libc++ to link to in the linker script.
    linked_libcxx = os.readlink(symlink_file)

    # Check that the abi_libname is one of the supported values.
    supported_abi_list = ['c++abi', 'stdc++', 'supc++', 'cxxrt']
    if abi_libname not in supported_abi_list:
        print_and_exit("abi name '%s' is not supported: Use one of %r" %
                        (abi_libname, supported_abi_list))

    # Generate the linker script contents and print the script and destination
    # information.
    contents = "INPUT(%s -l%s)" % (linked_libcxx, abi_libname)
    print("GENERATING SCRIPT: '%s' as file %s" % (contents, symlink_file))

    # Remove the existing libc++ symlink and replace it with the script.
    if not dryrun:
        os.unlink(symlink_file)
        with open(symlink_file, 'w') as f:
            f.write(contents + "\n")


if __name__ == '__main__':
    main()
