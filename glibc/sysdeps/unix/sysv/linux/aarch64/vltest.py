#!/usr/bin/python3
# Set Scalable Vector Length test helper
# Copyright (C) 2021 Free Software Foundation, Inc.
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
"""Set Scalable Vector Length test helper.

Set Scalable Vector Length for child process.

examples:

~/build$ make check subdirs=string \
test-wrapper='~/glibc/sysdeps/unix/sysv/linux/aarch64/vltest.py 16'

~/build$ ~/glibc/sysdeps/unix/sysv/linux/aarch64/vltest.py 16 \
make test t=string/test-memcpy

~/build$ ~/glibc/sysdeps/unix/sysv/linux/aarch64/vltest.py 32 \
./debugglibc.sh string/test-memmove

~/build$ ~/glibc/sysdeps/unix/sysv/linux/aarch64/vltest.py 64 \
./testrun.sh string/test-memset
"""
import argparse
from ctypes import cdll, CDLL
import os
import sys

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_UNSUPPORTED = 77

AT_HWCAP = 16
HWCAP_SVE = (1 << 22)

PR_SVE_GET_VL = 51
PR_SVE_SET_VL = 50
PR_SVE_SET_VL_ONEXEC = (1 << 18)
PR_SVE_VL_INHERIT = (1 << 17)
PR_SVE_VL_LEN_MASK = 0xffff

def main(args):
    libc = CDLL("libc.so.6")
    if not libc.getauxval(AT_HWCAP) & HWCAP_SVE:
        print("CPU doesn't support SVE")
        sys.exit(EXIT_UNSUPPORTED)

    libc.prctl(PR_SVE_SET_VL,
               args.vl[0] | PR_SVE_SET_VL_ONEXEC | PR_SVE_VL_INHERIT)
    os.execvp(args.args[0], args.args)
    print("exec system call failure")
    sys.exit(EXIT_FAILURE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
            "Set Scalable Vector Length test helper",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # positional argument
    parser.add_argument("vl", nargs=1, type=int,
                        choices=range(16, 257, 16),
                        help=('vector length '\
                              'which is multiples of 16 from 16 to 256'))
    # remainDer arguments
    parser.add_argument('args', nargs=argparse.REMAINDER,
                        help=('args '\
                              'which is passed to child process'))
    args = parser.parse_args()
    main(args)
