# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generic non-dexter-specific utility classes and functions."""

import os

from dex.utils.Environment import is_native_windows, has_pywin32
from dex.utils.PrettyOutputBase import PreserveAutoColors
from dex.utils.RootDirectory import get_root_directory
from dex.utils.Timer import Timer
from dex.utils.Warning import warn
from dex.utils.WorkingDirectory import WorkingDirectory

if is_native_windows():
    from dex.utils.windows.PrettyOutput import PrettyOutput
else:
    from dex.utils.posix.PrettyOutput import PrettyOutput
