# ====-- Type class for libc function headers -----------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

from hdrgen.symbol import Symbol


class Type(Symbol):
    # A type so far carries no specific information beyond its name.
    def __init__(self, name):
        super().__init__(name)
