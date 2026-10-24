# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

from . import PROGRAM


def report_note(verbose, message):
    if verbose:
        sys.stderr.write("%s: note: %s\n" % (PROGRAM, message))


def report_warning(message):
    sys.stderr.write("%s: warning: %s\n" % (PROGRAM, message))


def report_error(message, details=()):
    sys.stderr.write("%s: error: %s\n" % (PROGRAM, message))
    for detail in details:
        sys.stderr.write("  %s\n" % detail)


def report_info(key, value):
    sys.stdout.write("%s: %s\n" % (key, value))
    sys.stdout.flush()


def report_finding(text):
    sys.stderr.write(text + "\n")
