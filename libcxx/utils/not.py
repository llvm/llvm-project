#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

"""not.py is a utility for inverting the return code of commands.
It acts similar to llvm/utils/not.
ex: python /path/to/not.py ' echo hello
    echo $? // (prints 1)
"""

import subprocess
import sys

def which_cannot_find_program(prog):
    # Allow for import errors on distutils.spawn
    try:
        import distutils.spawn
        prog = distutils.spawn.find_executable(prog[0])
        if prog is None:
            sys.stderr.write('Failed to find program %s' % prog[0])
            return True
        return False
    except:
        return False

def main():
    argv = list(sys.argv)
    del argv[0]
    if len(argv) > 0 and argv[0] == '--crash':
        del argv[0]
        expectCrash = True
    else:
        expectCrash = False
    if len(argv) == 0:
        return 1
    if which_cannot_find_program(argv[0]):
        return 1
    rc = subprocess.call(argv)
    if rc < 0:
        return 0 if expectCrash else 1
    if expectCrash:
        return 1
    return rc == 0


if __name__ == '__main__':
    exit(main())
