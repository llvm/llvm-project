# ====- HeaderFile Class for libc function headers  -----------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

from functools import reduce
from pathlib import PurePosixPath


STDINT_SIZES = [
    "16",
    "32",
    "64",
    "8",
    "least16",
    "least32",
    "least64",
    "least8",
    "max",
    "ptr",
]

COMPILER_HEADER_TYPES = (
    {
        "bool": "<stdbool.h>",
        "va_list": "<stdarg.h>",
    }
    | {f"int{size}_t": "<stdint.h>" for size in STDINT_SIZES}
    | {f"uint{size}_t": "<stdint.h>" for size in STDINT_SIZES}
)


class HeaderFile:
    def __init__(self, name):
        self.template_file = None
        self.name = name
        self.macros = []
        self.types = []
        self.enumerations = []
        self.objects = []
        self.functions = []

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

    def all_types(self):
        return reduce(
            lambda a, b: a | b,
            [f.signature_types() for f in self.functions],
            set(self.types),
        )

    def includes(self):
        return {
            PurePosixPath("llvm-libc-macros") / macro.header
            for macro in self.macros
            if macro.header is not None
        } | {
            COMPILER_HEADER_TYPES.get(
                typ.type_name, PurePosixPath("llvm-libc-types") / f"{typ.type_name}.h"
            )
            for typ in self.all_types()
        }

    def public_api(self):
        # Python 3.12 has .relative_to(dir, walk_up=True) for this.
        path_prefix = PurePosixPath("../" * (len(PurePosixPath(self.name).parents) - 1))

        def relpath(file):
            return path_prefix / file

        content = [
            f"#include {file}"
            for file in sorted(
                file if isinstance(file, str) else f'"{relpath(file)!s}"'
                for file in self.includes()
            )
        ]

        for macro in self.macros:
            # When there is nothing to define, the Macro object converts to str
            # as an empty string.  Don't emit a blank line for those cases.
            if str(macro):
                content.extend(["", f"{macro}"])

        if self.enumerations:
            combined_enum_content = ",\n  ".join(
                str(enum) for enum in self.enumerations
            )
            content.append(f"\nenum {{\n  {combined_enum_content},\n}};")

        content.append("\n__BEGIN_C_DECLS\n")

        current_guard = None
        for function in self.functions:
            if function.guard == None:
                content.append(str(function) + " __NOEXCEPT;")
                content.append("")
            else:
                if current_guard == None:
                    current_guard = function.guard
                    content.append(f"#ifdef {current_guard}")
                    content.append(str(function) + " __NOEXCEPT;")
                    content.append("")
                elif current_guard == function.guard:
                    content.append(str(function) + " __NOEXCEPT;")
                    content.append("")
                else:
                    content.pop()
                    content.append(f"#endif // {current_guard}")
                    content.append("")
                    current_guard = function.guard
                    content.append(f"#ifdef {current_guard}")
                    content.append(str(function) + " __NOEXCEPT;")
                    content.append("")
        if current_guard != None:
            content.pop()
            content.append(f"#endif // {current_guard}")
            content.append("")

        content.extend(str(object) for object in self.objects)
        if self.objects:
            content.append("")
        content.append("__END_C_DECLS")

        return "\n".join(content)
