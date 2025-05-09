# ====-- Type class for libc function headers -----------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

from functools import total_ordering


@total_ordering
class Type:
    def __init__(self, type_name):
        assert type_name
        self.type_name = type_name

    def __eq__(self, other):
        return self.type_name == other.type_name

    def __lt__(self, other):
        return self.type_name < other.type_name

    def __hash__(self):
        return self.type_name.__hash__()
