# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Specializations for the Visual Studio 2015 interface."""

from dex.debugger.visualstudio.VisualStudio import VisualStudio


class VisualStudio2015(VisualStudio):
    @classmethod
    def get_name(cls):
        return 'Visual Studio 2015'

    @classmethod
    def get_option_name(cls):
        return 'vs2015'

    @property
    def _dte_version(self):
        return 'VisualStudio.DTE.14.0'
