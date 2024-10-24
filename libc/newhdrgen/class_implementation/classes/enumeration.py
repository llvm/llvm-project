# ====-- Enumeration class for libc function headers ----------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#


class Enumeration:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        if self.value != None:
            return f"{self.name} = {self.value}"
        else:
            return f"{self.name}"
