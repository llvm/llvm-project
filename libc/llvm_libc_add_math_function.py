#!/usr/bin/python
# Example usage:
#   cd path/to/llvm-project
#   ./path/to/llvm_libc_add_math_function.py '' 'TYPE' 'ceil' 'TYPE x' 'CeilTest' 'LIST_CEIL_TESTS'
#   ./path/to/llvm_libc_add_math_function.py 'f16' 'TYPE' 'ceil' 'TYPE x' 'CeilTest' 'LIST_CEIL_TESTS'
import sys
import subprocess

MAX_FILE_TITLE_LEN = 66

EMACS_CXX_MODE = "-*- C++ -*-"

FILE_HEADER_TEMPLATE = """//===-- {file_title}===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
"""

HEADER_TEMPLATE = """{file_header}
#ifndef LLVM_LIBC_SRC_MATH_{fn_identifier_uppercase}_H
#define LLVM_LIBC_SRC_MATH_{fn_identifier_uppercase}_H

#include "src/__support/macros/config.h"
{includes}
namespace LIBC_NAMESPACE_DECL {{

{fn_return_type} {fn_identifier}({fn_param_list});

}} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_{fn_identifier_uppercase}_H
"""

IMPL_TEMPLATE = """{file_header}
#include "src/math/{fn_identifier}.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {{

LLVM_LIBC_FUNCTION({fn_return_type}, {fn_identifier}, ({fn_param_list})) {{
  // TODO: Implement function.
}}

}} // namespace LIBC_NAMESPACE_DECL
"""

TEST_TEMPLATE = """{file_header}
#include "{test_class}.h"

#include "src/math/{fn_identifier}.h"

{test_macro}({fn_return_type}, LIBC_NAMESPACE::{fn_identifier})
"""


def get_type_from_suffix(suffix):
    match suffix:
        case "":
            return "double"
        case "f":
            return "float"
        case "l":
            return "long double"
        case "f16":
            return "float16"
        case "f128":
            return "float128"
        case _:
            raise ValueError("Unknown suffix")


def get_file_title(base_title, emacs_mode=""):
    dashes = "-" * (MAX_FILE_TITLE_LEN - len(base_title) - len(emacs_mode))
    return f"{base_title} {dashes}{emacs_mode}"


def get_include_for_type(type_identifier):
    match type_identifier:
        case "float16" | "float128":
            return '#include "src/__support/macros/properties/types.h"'


if __name__ == "__main__":
    (
        _,
        generic_type_suffix,
        fn_return_type_tmpl,
        fn_identifier_prefix,
        fn_param_list_tmpl,
        test_class,
        test_macro,
        *_,
    ) = sys.argv

    generic_type = get_type_from_suffix(generic_type_suffix)
    fn_return_type = fn_return_type_tmpl.replace("TYPE", generic_type)
    fn_identifier = fn_identifier_prefix + generic_type_suffix
    fn_identifier_uppercase = fn_identifier.upper()
    fn_param_list = fn_param_list_tmpl.replace("TYPE", generic_type)

    with open(f"libc/src/math/{fn_identifier}.h", "w") as header:
        header_title = get_file_title(
            f"Implementation header for {fn_identifier}", EMACS_CXX_MODE
        )
        header_file_header = FILE_HEADER_TEMPLATE.format(file_title=header_title)

        header_includes = ""

        if (generic_type_include := get_include_for_type(generic_type)) is not None:
            header_includes = f"{generic_type_include}\n"

        header.write(
            HEADER_TEMPLATE.format(
                file_header=header_file_header,
                fn_identifier_uppercase=fn_identifier_uppercase,
                includes=header_includes,
                fn_return_type=fn_return_type,
                fn_identifier=fn_identifier,
                fn_param_list=fn_param_list,
            )
        )

    with open(f"libc/src/math/generic/{fn_identifier}.cpp", "w") as impl:
        impl_title = get_file_title(f"Implementation of {fn_identifier} function")
        impl_file_header = FILE_HEADER_TEMPLATE.format(file_title=impl_title)

        impl.write(
            IMPL_TEMPLATE.format(
                file_header=impl_file_header,
                fn_return_type=fn_return_type,
                fn_identifier=fn_identifier,
                fn_param_list=fn_param_list,
            )
        )

    with open(f"libc/test/src/math/smoke/{fn_identifier}_test.cpp", "w") as test:
        test_title = get_file_title(f"Unittests for {fn_identifier}")
        test_file_header = FILE_HEADER_TEMPLATE.format(file_title=test_title)

        test.write(
            TEST_TEMPLATE.format(
                file_header=test_file_header,
                test_class=test_class,
                test_macro=test_macro,
                fn_return_type=fn_return_type,
                fn_identifier=fn_identifier,
            )
        )

    if generic_type == "float16":
        subprocess.run(
            r"sed -i 's/^\(| "
            + fn_identifier_prefix
            + r" \+\(| \(|check|\|N\/A\)\? \+\)\{3\}| \) \{7\}/\1|check|/' libc/docs/math/index.rst && git diff libc/docs/math/index.rst",
            shell=True,
            check=True,
        )