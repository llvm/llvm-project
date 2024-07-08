#!/usr/bin/env python
#
# ====-- Function class for libc function headers -------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#


class Function:
    def __init__(
        self, standards, return_type, name, arguments, guard=None, attributes=[]
    ):
        self.standards = standards
        self.return_type = return_type
        self.name = name
        self.arguments = [arg["type"] for arg in arguments]
        self.guard = guard
        self.attributes = attributes

    def __str__(self):
        args_str = ", ".join(self.arguments)
        attributes_str = " ".join(self.attributes)
        result = f"{self.return_type} {self.name}({args_str}){attributes_str};"
        if self.guard:
            result = f"#ifdef {self.guard}\n{result}\n#endif // {self.guard}"
        return result
