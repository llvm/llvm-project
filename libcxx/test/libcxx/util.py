#===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

from contextlib import contextmanager
import os
import tempfile


def cleanFile(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


@contextmanager
def guardedTempFilename(suffix='', prefix='', dir=None):
    # Creates and yeilds a temporary filename within a with statement. The file
    # is removed upon scope exit.
    handle, name = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    os.close(handle)
    yield name
    cleanFile(name)


@contextmanager
def guardedFilename(name):
    # yeilds a filename within a with statement. The file is removed upon scope
    # exit.
    yield name
    cleanFile(name)


@contextmanager
def nullContext(value):
    # yeilds a variable within a with statement. No action is taken upon scope
    # exit.
    yield value


def makeReport(cmd, out, err, rc):
    report = "Command: %s\n" % cmd
    report += "Exit Code: %d\n" % rc
    if out:
        report += "Standard Output:\n--\n%s--\n" % out
    if err:
        report += "Standard Error:\n--\n%s--\n" % err
    report += '\n'
    return report
