#!/usr/bin/env python

"""
A simple utility to redo the failed/errored tests.

You need to specify the session directory in order for this script to locate the
tests which need to be re-run.

See also dotest.py, the test driver running the test suite.

Type:

./dotest.py -h

for help.
"""

from __future__ import print_function

import os
import sys
import datetime
import re

# If True, redo with no '-t' option for the test driver.
no_trace = False

# To be filled with the filterspecs found in the session logs.
redo_specs = []

# The filename components to match for.  Only files with the contained component names
# will be considered for re-run.  Examples: ['X86_64', 'clang'].
filename_components = []

do_delay = False

# There is a known bug with respect to comp_specs and arch_specs, in that if we
# encountered "-C clang" and "-C gcc" when visiting the session files, both
# compilers will end up in the invocation of the test driver when rerunning.
# That is: ./dotest -v -C clang^gcc ... -f ...".  Ditto for "-A" flags.

# The "-C compiler" for comp_specs.
comp_specs = set()
# The "-A arch" for arch_specs.
arch_specs = set()


def usage():
    print("""\
Usage: redo.py [-F filename_component] [-n] [session_dir] [-d]
where options:
-F : only consider the test for re-run if the session filename contains the filename component
     for example: -F x86_64
-n : when running the tests, do not turn on trace mode, i.e, no '-t' option
     is passed to the test driver (this will run the tests faster)
-d : pass -d down to the test driver (introduces a delay so you can attach with a debugger)

and session_dir specifies the session directory which contains previously
recorded session infos for all the test cases which either failed or errored.

If sessin_dir is left unspecified, this script uses the heuristic to find the
possible session directories with names starting with %Y-%m-%d- (for example,
2012-01-23-) and employs the one with the latest timestamp.""")
    sys.exit(0)


def where(session_dir, test_dir):
    """Returns the full path to the session directory; None if non-existent."""
    abspath = os.path.abspath(session_dir)
    if os.path.isdir(abspath):
        return abspath

    session_dir_path = os.path.join(test_dir, session_dir)
    if os.path.isdir(session_dir_path):
        return session_dir_path

    return None

# This is the pattern for the line from the log file to redo a test.
# We want the filter spec.
filter_pattern = re.compile("^\./dotest\.py.*-f (.*)$")
comp_pattern = re.compile(" -C ([^ ]+) ")
arch_pattern = re.compile(" -A ([^ ]+) ")


def redo(suffix, dir, names):
    """Visitor function for os.path.walk(path, visit, arg)."""
    global redo_specs
    global comp_specs
    global arch_specs
    global filter_pattern
    global comp_pattern
    global arch_pattern
    global filename_components
    global do_delay

    for name in names:
        if name.endswith(suffix):
            #print("Find a log file:", name)
            if name.startswith("Error") or name.startswith("Failure"):
                if filename_components:
                    if not all([comp in name for comp in filename_components]):
                        continue
                with open(os.path.join(dir, name), 'r') as log:
                    content = log.read()
                    for line in content.splitlines():
                        match = filter_pattern.match(line)
                        if match:
                            filterspec = match.group(1)
                            print("adding filterspec:", filterspec)
                            redo_specs.append(filterspec)
                            comp = comp_pattern.search(line)
                            if comp:
                                comp_specs.add(comp.group(1))
                            arch = arch_pattern.search(line)
                            if arch:
                                arch_specs.add(arch.group(1))
            else:
                continue


def main():
    """Read the session directory and run the failed test cases one by one."""
    global no_trace
    global redo_specs
    global filename_components
    global do_delay

    test_dir = sys.path[0]
    if not test_dir:
        test_dir = os.getcwd()
    if not test_dir.endswith('test'):
        print("This script expects to reside in lldb's test directory.")
        sys.exit(-1)

    index = 1
    while index < len(sys.argv):
        if sys.argv[index].startswith(
                '-h') or sys.argv[index].startswith('--help'):
            usage()

        if sys.argv[index].startswith('-'):
            # We should continue processing...
            pass
        else:
            # End of option processing.
            break

        if sys.argv[index] == '-F':
            # Increment by 1 to fetch the filename component spec.
            index += 1
            if index >= len(sys.argv) or sys.argv[index].startswith('-'):
                usage()
            filename_components.append(sys.argv[index])
        elif sys.argv[index] == '-n':
            no_trace = True
        elif sys.argv[index] == '-d':
            do_delay = True

        index += 1

    if index < len(sys.argv):
        # Get the specified session directory.
        session_dir = sys.argv[index]
    else:
        # Use heuristic to find the latest session directory.
        name = datetime.datetime.now().strftime("%Y-%m-%d-")
        dirs = [d for d in os.listdir(os.getcwd()) if d.startswith(name)]
        if len(dirs) == 0:
            print("No default session directory found, please specify it explicitly.")
            usage()
        session_dir = max(dirs, key=os.path.getmtime)
        if not session_dir or not os.path.exists(session_dir):
            print("No default session directory found, please specify it explicitly.")
            usage()

    #print("The test directory:", test_dir)
    session_dir_path = where(session_dir, test_dir)

    print("Using session dir path:", session_dir_path)
    os.chdir(test_dir)
    os.path.walk(session_dir_path, redo, ".log")

    if not redo_specs:
        print("No failures/errors recorded within the session directory, please specify a different session directory.\n")
        usage()

    filters = " -f ".join(redo_specs)
    compilers = ''
    for comp in comp_specs:
        compilers += " -C %s" % (comp)
    archs = ''
    for arch in arch_specs:
        archs += "--arch %s " % (arch)

    command = "./dotest.py %s %s -v %s %s -f " % (
        compilers, archs, "" if no_trace else "-t", "-d" if do_delay else "")

    print("Running %s" % (command + filters))
    os.system(command + filters)

if __name__ == '__main__':
    main()
