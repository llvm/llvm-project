import os
import sys
import argparse
import re

class Checks(object):
    class CheckError(Exception):
        pass

    def __init__(self, filename, prefix):
        self.checks = []
        self.lines = []
        self.check_no_output = False
        self.filename = filename
        self.prefix = prefix
    def readStdin(self):
        self.lines = [l.rstrip('\r\n') for l in sys.stdin.readlines()]
    def readChecks(self):
        with open(self.filename) as f:
            for line in f:
                match = re.search('{}: NO_OUTPUT'.format(self.prefix), line)
                if match is not None:
                    self.check_no_output = True
                    return
                match = re.search('{}: num_threads=([0-9]+) (.*)$'.format(self.prefix), line)
                if match is not None:
                    num_threads = int(match.group(1))
                    for i in range(num_threads):
                        self.checks.append(match.group(2))
                    continue
    def check(self):
        # If no checks at all, then nothing to do
        if len(self.checks) == 0 and not self.check_no_output:
            print('Nothing to check for')
            return
        # Check if we are expecting no output
        if self.check_no_output:
            if len(self.lines) == 0:
                return
            else:
                raise Checks.CheckError('{}: Output was found when expecting none.'.format(self.prefix))
        # Run through each check line and see if it exists in the output
        # If it does, then delete the line from output and look for the
        # next check line.
        # If you don't find the line then raise Checks.CheckError
        # If there are extra lines of output then raise Checks.CheckError
        for c in self.checks:
            found = False
            index = -1
            for idx, line in enumerate(self.lines):
                if re.search(c, line) is not None:
                    found = True
                    index = idx
                    break
            if not found:
                raise Checks.CheckError('{}: Did not find: {}'.format(self.prefix, c))
            else:
                del self.lines[index]
        if len(self.lines) != 0:
            raise Checks.CheckError('{}: Extra output: {}'.format(self.prefix, self.lines))

# Setup argument parsing
parser = argparse.ArgumentParser(description='''This script checks output of
    a program against "CHECK" lines in filename''')
parser.add_argument('filename', default=None, help='filename to check against')
parser.add_argument('-c', '--check-prefix', dest='prefix',
                    default='CHECK', help='check prefix token default: %(default)s')
command_args = parser.parse_args()
# Do the checking
checks = Checks(command_args.filename, command_args.prefix)
checks.readStdin()
checks.readChecks()
checks.check()
