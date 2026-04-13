# ====-- Object class for libc function headers ---------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

from hdrgen.symbol import Symbol


class Object(Symbol):
    def __init__(self, name, type):
        super().__init__(name)
        self.type = type

    def __str__(self):
        return f"extern {self.type} {self.name};"
