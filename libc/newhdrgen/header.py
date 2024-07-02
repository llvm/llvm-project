#!/usr/bin/env python
#
# ====- HeaderFile Class for libc function headers  -----------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#


class HeaderFile:
    def __init__(self, name):
        self.name = name
        self.macros = []
        self.types = []
        self.enumerations = []
        self.objects = []
        self.functions = []
        self.includes = []

    def add_macro(self, macro):
        self.macros.append(macro)

    def add_type(self, type_):
        self.types.append(type_)

    def add_enumeration(self, enumeration):
        self.enumerations.append(enumeration)

    def add_object(self, object):
        self.objects.append(object)

    def add_function(self, function):
        self.functions.append(function)

    def add_include(self, include):
        self.includes.append(include)

    def __str__(self):
        content = [""]

        for include in self.includes:
            content.append(str(include))

        for macro in self.macros:
            content.append(str(macro))

        for object in self.objects:
            content.append(str(object))

        for type_ in self.types:
            content.append(str(type_))

        if self.enumerations:
            content.append("enum {")
            for enum in self.enumerations:
                content.append(f"\t{str(enum)},")
            content.append("};")

        # TODO: replace line below with common.h functionality
        content.append("__BEGIN_C_DECLS\n")
        for function in self.functions:
            content.append(str(function))
            content.append("")
        content.append("__END_C_DECLS\n")
        return "\n".join(content)
