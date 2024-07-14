#!/usr/bin/env python
#
# ===- HeaderFile Class for libc function headers  -----------*- python -*--==#
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
        content = []

        content.append(
            f"//===-- C standard declarations for {self.name} ------------------------------===//"
        )
        content.append("//")
        content.append(
            "// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions."
        )
        content.append("// See https://llvm.org/LICENSE.txt for license information.")
        content.append("// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception")
        content.append("//")
        content.append(
            "//===----------------------------------------------------------------------===//\n"
        )

        header_guard = f"__LLVM_LIBC_DECLARATIONS_{self.name.upper()[:-2]}_H"
        content.append(f"#ifndef {header_guard}")
        content.append(f"#define {header_guard}\n")

        content.append("#ifndef __LIBC_ATTRS")
        content.append("#define __LIBC_ATTRS")
        content.append("#endif\n")

        content.append("#ifdef __cplusplus")
        content.append('extern "C" {')
        content.append("#endif\n")

        for include in self.includes:
            content.append(str(include))

        for macro in self.macros:
            content.append(f"{macro}\n")

        for type_ in self.types:
            content.append(f"{type_}")

        if self.enumerations:
            combined_enum_content = ",\n  ".join(
                str(enum) for enum in self.enumerations
            )
            content.append(f"\nenum {{\n  {combined_enum_content},\n}};")

        for function in self.functions:
            content.append(f"{function} __LIBC_ATTRS;\n")

        for object in self.objects:
            content.append(f"{object} __LIBC_ATTRS;\n")

        content.append("#ifdef __cplusplus")
        content.append("}")
        content.append("#endif\n")

        content.append(f"#endif")

        return "\n".join(content)
