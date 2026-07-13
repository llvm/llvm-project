# ====-- Enumeration class for libc function headers ----------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

from hdrgen.symbol import Symbol


class Enumeration(Symbol):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value

    def __str__(self):
        if self.value != None:
            return f"{self.name} = {self.value}"
        else:
            return f"{self.name}"
