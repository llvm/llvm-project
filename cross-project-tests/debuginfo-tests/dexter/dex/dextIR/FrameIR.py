# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from dex.dextIR.LocIR import LocIR


class FrameIR:
    """Data class which represents a frame in the call stack"""

    def __init__(self, function: str, is_inlined: bool, loc: LocIR):
        self.function = function
        self.is_inlined = is_inlined
        self.loc = loc
