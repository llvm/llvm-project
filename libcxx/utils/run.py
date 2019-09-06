#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

"""run.py is a utility for running a program.

It can perform code signing, forward arguments to the program, and return the
program's error code.
"""

import subprocess
import sys


def main():
    codesign_ident = sys.argv[1]

    # Ignore 'run.py' and the codesigning identity.
    argv = sys.argv[2:]

    exec_path = argv[0]

    # Do any necessary codesigning.
    if codesign_ident:
        sign_cmd = ['xcrun', 'codesign', '-f', '-s', codesign_ident, exec_path]
        cs_rc = subprocess.call(sign_cmd, env={})
        if cs_rc != 0:
            sys.stderr.write('Failed to codesign: ' + exec_path)
            return cs_rc

    return subprocess.call(argv)

if __name__ == '__main__':
    exit(main())
