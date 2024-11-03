# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utility functions for querying the current environment."""

import os


def is_native_windows():
    return os.name == "nt"


def has_pywin32():
    try:
        import win32com.client  # pylint:disable=unused-variable
        import win32api  # pylint:disable=unused-variable

        return True
    except ImportError:
        return False
