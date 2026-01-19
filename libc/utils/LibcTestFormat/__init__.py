# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

"""
Lit test format for LLVM libc unit tests.

This format extends lit.formats.ExecutableTest to discover pre-built test
executables in the build directory. Test executables are expected to follow
the naming pattern used by add_libc_test():
  libc.test.src.<category>.<test_name>.__unit__.__build__
  libc.test.src.<category>.<test_name>.__hermetic__.__build__
"""

from .format import LibcTest

__all__ = ["LibcTest"]
