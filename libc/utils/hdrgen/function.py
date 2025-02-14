# ====-- Function class for libc function headers -------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

import re
from type import Type


# These are the keywords that appear in C type syntax but are not part of the
# include file name.  This is all of the modifiers, qualifiers, and base types,
# but not "struct".
KEYWORDS = [
    "_Atomic",
    "_Complex",
    "_Float16",
    "_Noreturn",
    "__restrict",
    "accum",
    "char",
    "const",
    "double",
    "float",
    "fract",
    "int",
    "long",
    "short",
    "signed",
    "unsigned",
    "void",
    "volatile",
]
NONIDENTIFIER = re.compile("[^a-zA-Z0-9_]+")


class Function:
    def __init__(
        self, return_type, name, arguments, standards, guard=None, attributes=[]
    ):
        assert return_type
        self.return_type = return_type
        self.name = name
        self.arguments = [
            arg if isinstance(arg, str) else arg["type"] for arg in arguments
        ]
        assert all(self.arguments)
        self.standards = standards
        self.guard = guard
        self.attributes = attributes or []

    def signature_types(self):
        def collapse(type_string):
            assert type_string
            # Split into words at nonidentifier characters (`*`, `[`, etc.),
            # filter out keywords and numbers, and then rejoin with "_".
            return "_".join(
                word
                for word in NONIDENTIFIER.split(type_string)
                if word and not word.isdecimal() and word not in KEYWORDS
            )

        all_types = [self.return_type] + self.arguments
        return {
            Type(string) for string in filter(None, (collapse(t) for t in all_types))
        }

    def __str__(self):
        attrs_str = "".join(f"{attr} " for attr in self.attributes)
        arguments_str = ", ".join(self.arguments) if self.arguments else "void"
        return attrs_str + f"{self.return_type} {self.name}({arguments_str})"
