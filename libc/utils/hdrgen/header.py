# ====- HeaderFile Class for libc function headers  -----------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

from pathlib import PurePath


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

    def includes(self):
        return sorted(
            {
                PurePath("llvm-libc-macros") / macro.header
                for macro in self.macros
                if macro.header is not None
            }
        )

    def public_api(self):
        header_dir = PurePath(self.name).parent
        content = [
            f'#include "{file.relative_to(header_dir)}"' for file in self.includes()
        ] + [""]

        for macro in self.macros:
            # When there is nothing to define, the Macro object converts to str
            # as an empty string.  Don't emit a blank line for those cases.
            if str(macro):
                content.append(f"{macro}\n")

        for type_ in self.types:
            content.append(f"{type_}")

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
