# ====-- Function class for libc function headers -------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

import re
from functools import total_ordering
from hdrgen.type import Type


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


@total_ordering
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

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return self.name.__hash__()

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
        # The rendering of the return type may look like `int` or it may look
        # like `int *` (and other examples).  For `int`, a space is always
        # needed to separate the tokens.  For `int *`, no whitespace matters to
        # the syntax one way or the other, but an extra space between `*` and
        # the function identifier is not the canonical style.
        type_str = str(self.return_type)
        if type_str[-1].isalnum() or type_str[-1] == "_":
            type_str += " "
        return attrs_str + type_str + self.name + "(" + arguments_str + ")"
