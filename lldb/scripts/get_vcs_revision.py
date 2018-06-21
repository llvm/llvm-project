#!/usr/bin/env python
"""
Retrieve the Version Control System revision number.

This script checks the current directory, figures out
what kind of revision control system it is (currently
supports git and svn), and reports the revision number
if it can figure out what it is.  If it cannot figure
out what it is, it prints nothing to standard out.
"""

from __future__ import print_function

import os
import subprocess


def print_git_revision():
    try:
        revision = subprocess.check_output('git rev-parse HEAD', shell=True)
        print(revision.rstrip())
    except:
        # Print nothing on error.
        pass


def print_svn_revision():
    try:
        revision = subprocess.check_output('svnversion', shell=True)
        print(revision.rstrip())
    except:
        # Print nothing on error.
        pass


def main():
    """Drives the main program."""
    if os.path.exists(".git"):
        print_git_revision()
    elif os.path.exists(".svn"):
        print_svn_revision()
    else:
        # VCS not yet supported.
        pass

if __name__ == "__main__":
    main()
