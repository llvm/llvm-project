# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

PROGRAM = "clang-ssaf-driver"

try:
    from .version import VERSION
except ImportError:
    VERSION = "unknown (not built)"
