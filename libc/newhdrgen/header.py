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

    def __str__(self):
        content = [""]

        for macro in self.macros:
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

        for object in self.objects:
            content.append(str(object))
        if self.objects:
            content.append("\n__END_C_DECLS")
        else:
            content.append("__END_C_DECLS")

        return "\n".join(content)
