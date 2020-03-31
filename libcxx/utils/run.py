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

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--codesign_identity', type=str, required=False)
    parser.add_argument('--working_directory', type=str, required=True)
    parser.add_argument('--dependencies', type=str, nargs='*', required=True)
    parser.add_argument('--env', type=str, nargs='*', required=True)
    (args, remaining) = parser.parse_known_args(sys.argv[1:])

    if len(remaining) < 2:
        sys.stderr.write('Missing actual commands to run')
        exit(1)
    remaining = remaining[1:] # Skip the '--'

    # Do any necessary codesigning.
    if args.codesign_identity:
        exe = remaining[0]
        rc = subprocess.call(['xcrun', 'codesign', '-f', '-s', args.codesign_identity, exe], env={})
        if rc != 0:
            sys.stderr.write('Failed to codesign: ' + exe)
            return rc

    # Extract environment variables into a dictionary
    env = {k : v  for (k, v) in map(lambda s: s.split('=', 1), args.env)}

    # Ensure the file dependencies exist
    for file in args.dependencies:
        if not os.path.exists(file):
            sys.stderr.write('Missing file {} marked as a dependency of a test'.format(file))
            exit(1)

    # Run the executable with the given environment in the given working directory
    return subprocess.call(' '.join(remaining), cwd=args.working_directory, env=env, shell=True)

if __name__ == '__main__':
    exit(main())
