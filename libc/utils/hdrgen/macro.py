# ====-- Macro class for libc function headers ----------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#


class Macro:
    def __init__(self, name, value=None, header=None):
        self.name = name
        self.value = value
        self.header = header

    def __str__(self):
        if self.header != None:
            return f"""#ifndef {self.name}
#error "{self.name} should be defined by llvm-libc-macros/{self.header}"
#endif"""
        if self.value != None:
            return f"#define {self.name} {self.value}"
        return f"#define {self.name}"
