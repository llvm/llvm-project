"""
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Distinguishes the two reasons a test can end up not running.
"""


class UnsupportedReason(str):
    """A skip reason meaning "this test can never run here", not "this test is
    broken here". Reported as UNSUPPORTED rather than SKIPPED.

    *message* says what the decorator requires ("requires one of darwin").
    *reason* is the optional test-specific explanation the decorator was handed
    ("uses Darwin APIs"); it is appended to *message* so both end up in the
    test report.
    """

    def __new__(cls, message, reason=None):
        assert isinstance(
            reason, (str, type(None))
        ), f"expects 'str' or 'None' got {type(reason).__name__!r}"
        if reason:
            message = f"{message}: {reason}"
        return super().__new__(cls, message)


def is_unsupported(reason):
    """Return True if *reason* marks a test as unsupported rather than skipped."""
    return isinstance(reason, UnsupportedReason)
