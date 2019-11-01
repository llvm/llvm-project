# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utility functions related to DExTer's directory layout."""

import os


def get_root_directory():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    assert os.path.basename(root) == 'dex', root
    return root
