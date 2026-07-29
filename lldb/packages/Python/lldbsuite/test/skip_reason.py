"""
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Distinguishes the two reasons a test can end up not running.

The test suite skips tests for two fundamentally different reasons:

  * The test *cannot* run in this configuration and never will.  A test for
    Mach-O debug maps has nothing to say on Linux; a test that forks has
    nothing to say on Windows.  These are reported as **UNSUPPORTED**.

  * The test *should* run in this configuration but doesn't work yet.  Someone
    has to fix it.  These are reported as **SKIPPED**.

Both go through `unittest`'s skip machinery, so the distinction is carried by
the type of the reason string: decorators that express a hard requirement pass
an `UnsupportedReason` instead of a plain `str`.  `unittest` hands the reason
object to `TestResult.addSkip` untouched, which lets `LLDBTestResult` tell the
two apart.

Note that `TestCase.skipTest()` stringifies its argument, which loses the
marker.  Decorators that need to report UNSUPPORTED must therefore go through
`unittest.skipIf` / `unittest.skipUnless` rather than `skipTestIfFn`.
"""


class UnsupportedReason(str):
    """A skip reason meaning "this test can never run here", not "this test is
    broken here". Reported as UNSUPPORTED rather than SKIPPED."""


def is_unsupported(reason):
    """Return True if *reason* marks a test as unsupported rather than skipped."""
    return isinstance(reason, UnsupportedReason)
