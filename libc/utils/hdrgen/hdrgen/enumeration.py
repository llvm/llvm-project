# ====-- Enumeration class for libc function headers ----------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

from functools import total_ordering


@total_ordering
class Enumeration:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        if self.value != None:
            return f"{self.name} = {self.value}"
        else:
            return f"{self.name}"
