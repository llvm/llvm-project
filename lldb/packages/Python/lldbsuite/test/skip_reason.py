"""
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Distinguishes the two reasons a test can end up not running.
"""


class UnsupportedReason(str):
    """A skip reason meaning "this test can never run here", not "this test is
    broken here". Reported as UNSUPPORTED rather than SKIPPED."""


def is_unsupported(test, reason):
    """Return True if *reason* marks a test as unsupported rather than skipped."""
    return isinstance(reason, UnsupportedReason) or getattr(
        test, "_skipped_as_unsupported", False
    )
