# ====- HeaderFile Class for libc function headers  -----------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

import re
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

COMPILER_HEADER_TYPES = {
    "bool": "<stdbool.h>",
    "va_list": "<stdarg.h>",
}
COMPILER_HEADER_TYPES.update({f"int{size}_t": "<stdint.h>" for size in STDINT_SIZES})
COMPILER_HEADER_TYPES.update({f"uint{size}_t": "<stdint.h>" for size in STDINT_SIZES})

NONIDENTIFIER = re.compile("[^a-zA-Z0-9_]+")

COMMON_HEADER = PurePosixPath("__llvm-libc-common.h")

# All the canonical identifiers are in lowercase for easy maintenance.
# This maps them to the pretty descriptions to generate in header comments.
LIBRARY_DESCRIPTIONS = {
    "stdc": "Standard C",
    "posix": "POSIX",
    "bsd": "BSD",
    "gnu": "GNU",
    "linux": "Linux",
    "uefi": "UEFI",
    "svid": "SVID",
}

HEADER_TEMPLATE = """\
//===-- {library} header <{header}> --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef {guard}
#define {guard}

%%public_api()

#endif // {guard}
"""


class HeaderFile:
    def __init__(self, name):
        self.template_file = None
        self.name = name
        self.macros = []
        self.types = []
        self.enumerations = []
        self.objects = []
        self.functions = []
        self.standards = []
        self.merge_yaml_files = []

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

    def merge(self, other):
        self.macros = sorted(set(self.macros) | set(other.macros))
        self.types = sorted(set(self.types) | set(other.types))
        self.enumerations = sorted(set(self.enumerations) | set(other.enumerations))
        self.objects = sorted(set(self.objects) | set(other.objects))
        self.functions = sorted(set(self.functions) | set(other.functions))

    def all_types(self):
        return reduce(
            lambda a, b: a | b,
            [f.signature_types() for f in self.functions],
            set(self.types),
        )

    def all_standards(self):
        # FIXME: Only functions have the "standard" field, but all the entity
        # types should have one too.
        return set(self.standards).union(
            *(filter(None, (f.standards for f in self.functions)))
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

    def header_guard(self):
        return "_LLVM_LIBC_" + "_".join(
            word.upper() for word in NONIDENTIFIER.split(self.name) if word
        )

    def library_description(self):
        # If the header itself is in standard C, just call it that.
        if "stdc" in self.standards:
            return LIBRARY_DESCRIPTIONS["stdc"]
        # If the header itself is in POSIX, just call it that.
        if "posix" in self.standards:
            return LIBRARY_DESCRIPTIONS["posix"]
        # Otherwise, consider the standards for each symbol as well.
        standards = self.all_standards()
        # Otherwise, it's described by all those that apply, but ignoring
        # "stdc" and "posix" since this is not a "stdc" or "posix" header.
        return " / ".join(
            sorted(
                LIBRARY_DESCRIPTIONS[standard]
                for standard in standards
                if standard not in {"stdc", "posix"}
            )
        )

    def template(self, dir, files_read):
        if self.template_file is not None:
            # There's a custom template file, so just read it in and record
            # that it was read as an input file.
            template_path = dir / self.template_file
            files_read.add(template_path)
            return template_path.read_text()

        # Generate the default template.
        return HEADER_TEMPLATE.format(
            library=self.library_description(),
            header=self.name,
            guard=self.header_guard(),
        )

    def public_api(self):
        # Python 3.12 has .relative_to(dir, walk_up=True) for this.
        path_prefix = PurePosixPath("../" * (len(PurePosixPath(self.name).parents) - 1))

        def relpath(file):
            return path_prefix / file

        content = []

        if self.template_file is None:
            # This always goes before all the other includes, which are sorted.
            # It's implicitly emitted here when using the default template so
            # it can get the right relative path.  Custom template files should
            # all have it explicitly with their right particular relative path.
            content.append('#include "{file!s}"'.format(file=relpath(COMMON_HEADER)))

        content += [
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
            if function.guard == None and current_guard == None:
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
                    if current_guard is not None:
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

    def json_data(self):
        return {
            "name": self.name,
            "standards": self.standards,
            "includes": [
                str(file) for file in sorted({COMMON_HEADER} | self.includes())
            ],
        }
