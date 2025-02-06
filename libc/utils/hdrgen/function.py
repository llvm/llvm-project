# ====-- Function class for libc function headers -------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#


class Function:
    def __init__(
        self, return_type, name, arguments, standards, guard=None, attributes=[]
    ):
        self.return_type = return_type
        self.name = name
        self.arguments = [
            arg if isinstance(arg, str) else arg["type"] for arg in arguments
        ]
        self.standards = standards
        self.guard = guard
        self.attributes = attributes or ""

    def __str__(self):
        attributes_str = " ".join(self.attributes)
        arguments_str = ", ".join(self.arguments) if self.arguments else "void"
        if attributes_str == "":
            result = f"{self.return_type} {self.name}({arguments_str})"
        else:
            result = f"{attributes_str} {self.return_type} {self.name}({arguments_str})"
        return result
