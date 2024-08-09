# ====-- Type class for libc function headers -----------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#


class Type:
    def __init__(self, type_name):
        self.type_name = type_name

    def __str__(self):
        return f"#include <llvm-libc-types/{self.type_name}.h>"
