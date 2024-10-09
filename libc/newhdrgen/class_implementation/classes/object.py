# ====-- Object class for libc function headers ---------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#


class Object:
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def __str__(self):
        return f"extern {self.type} {self.name};"
