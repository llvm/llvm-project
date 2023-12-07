# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class BuilderIR:
    """Data class which represents the compiler related options passed to Dexter"""

    def __init__(self, name: str, cflags: str, ldflags: str):
        self.name = name
        self.cflags = cflags
        self.ldflags = ldflags
