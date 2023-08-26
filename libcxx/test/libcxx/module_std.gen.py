# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# Test that all named declarations with external linkage match the
# exported declarations in their associated module partition.
# Then it tests the sum of the exported declarations in the module
# partitions matches the export of the std module.

# Note the test of the std module requires all partitions to be tested
# first. Since lit tests have no dependencies, this means the test needs
# to be one monolitic test. Since the test doesn't take very long it's
# not a huge issue.

# RUN: %{python} %s %{libcxx}/utils

import sys

sys.path.append(sys.argv[1])
from libcxx.header_information import module_headers
from libcxx.header_information import header_restrictions

BLOCKLIT = (
    ""  # block Lit from interpreting a RUN/XFAIL/etc inside the generation script
)

# Ignore several declarations found in the includes.
#
# Part of these items are bugs other are not yet implemented features.
SkipDeclarations = dict()

# See comment in the header.
SkipDeclarations["cuchar"] = ["std::mbstate_t", "std::size_t"]

# Not in the synopsis.
SkipDeclarations["cwchar"] = ["std::FILE"]

# The operators are added for private types like __iom_t10.
SkipDeclarations["iomanip"] = ["std::operator<<", "std::operator>>"]

SkipDeclarations["iosfwd"] = ["std::ios_base", "std::vector"]

# This header also provides declarations in the namespace that might be
# an error.
SkipDeclarations["filesystem"] = [
    "std::filesystem::operator==",
    "std::filesystem::operator!=",
]

# This is a specialization for a private type
SkipDeclarations["iterator"] = ["std::pointer_traits"]

# TODO MODULES
# This definition is declared in string and defined in istream
# This declaration should be part of string
SkipDeclarations["istream"] = ["std::getline"]

# P1614 (at many places) and LWG3519 too.
SkipDeclarations["random"] = [
    "std::operator!=",
    # LWG3519 makes these hidden friends.
    # Note the older versions had the requirement of these operations but not in
    # the synopsis.
    "std::operator<<",
    "std::operator>>",
    "std::operator==",
]

# Declared in the forward header since std::string uses std::allocator
SkipDeclarations["string"] = ["std::allocator"]
# TODO MODULES remove zombie names
# https://libcxx.llvm.org/Status/Cxx20.html#note-p0619
SkipDeclarations["memory"] = [
    "std::return_temporary_buffer",
    "std::get_temporary_buffer",
]

# TODO MODULES this should be part of ios instead
SkipDeclarations["streambuf"] = ["std::basic_ios"]

# include/__type_traits/is_swappable.h
SkipDeclarations["type_traits"] = [
    "std::swap",
    # TODO MODULES gotten through __functional/unwrap_ref.h
    "std::reference_wrapper",
]

# Add declarations in headers.
#
# Some headers have their defines in a different header, which may have
# additional declarations.
ExtraDeclarations = dict()
# This declaration is in the ostream header.
ExtraDeclarations["system_error"] = ["std::operator<<"]

# Adds an extra header file to scan
#
#
ExtraHeader = dict()
# locale has a file and not a subdirectory
ExtraHeader["locale"] = "v1/__locale$"
ExtraHeader["thread"] = "v1/__threading_support$"
ExtraHeader["ranges"] = "v1/__fwd/subrange.h$"

# The extra header is needed since two headers are required to provide the
# same definition.
ExtraHeader["functional"] = "v1/__compare/compare_three_way.h$"

# newline needs to be escaped for the module partition output.
nl = '\\\\n'

# Create empty file with all parts.
print(
    f"""\
//--- module_std.sh.cpp
// UNSUPPORTED{BLOCKLIT}: c++03, c++11, c++14, c++17
// UNSUPPORTED{BLOCKLIT}: libcpp-has-no-std-modules
// UNSUPPORTED{BLOCKLIT}: clang-modules-build

// REQUIRES{BLOCKLIT}: has-clang-tidy

// The GCC compiler flags are not always compatible with clang-tidy.
// UNSUPPORTED{BLOCKLIT}: gcc

// RUN{BLOCKLIT}: echo -n > %t.all_partitions
"""
)

# Validate all module parts.
for header in module_headers:
    # Some headers cannot be included when a libc++ feature is disabled.
    # In that case include the header conditionally. The header __config
    # ensures the libc++ feature macros are available.
    if header in header_restrictions:
        include = (
            f"#include <__config>{nl}"
            + f"#if {header_restrictions[header]}{nl}"
            + f"#  include <{header}>{nl}"
            + f"#endif{nl}"
        )
    elif header == "chrono":
        # When localization is disabled the header string is not included.
        # When string is included chrono's operator""s is a named declaration
        #   using std::chrono_literals::operator""s;
        # else it is a named declaration
        #   using std::operator""s;
        # TODO MODULES investigate why
        include = f"#include <string>{nl}#include <chrono>{nl}"
    else:
        include = f"#include <{header}>{nl}"

    # Generate a module partition for the header module includes. This
    # makes it possible to verify that all headers export all their
    # named declarations.
    print(
        f"// RUN{BLOCKLIT}: echo -e \""
        f"module;{nl}"
        f"{include}"
        f"{nl}"
        f"// Use __libcpp_module_<HEADER> to ensure that modules {nl}"
        f"// are not named as keywords or reserved names.{nl}"
        f"export module std:__libcpp_module_{header};{nl}"
        f'#include \\"%{{module}}/std/{header}.inc\\"{nl}'
        f"\" > %t.{header}.cppm")

    # Dump the information as found in the module's cppm file.
    print(
        f"// RUN{BLOCKLIT}: %{{clang-tidy}} %t.{header}.cppm "
        "  --checks='-*,libcpp-header-exportable-declarations' "
        "  -config='{CheckOptions: [ "
        "    {"
        "      key: libcpp-header-exportable-declarations.Filename, "
        f"     value: {header}.inc"
        "    }, {"
        "      key: libcpp-header-exportable-declarations.FileType, "
        "      value: ModulePartition"
        "    }, "
        "  ]}' "
        "  --load=%{test-tools}/clang_tidy_checks/libcxx-tidy.plugin "
        "  -- %{flags} %{compile_flags} "
        f"| sort > %t.{header}.module"
    )
    print(f"// RUN{BLOCKLIT}: cat  %t.{header}.module >> %t.all_partitions")

    # Dump the information as found in the module by using the header file(s).
    skip_declarations = " ".join(SkipDeclarations.get(header, []))
    if skip_declarations:
        skip_declarations = (
            "{"
            "  key: libcpp-header-exportable-declarations.SkipDeclarations, "
            f' value: "{skip_declarations}" '
            "}, "
        )

    extra_declarations = " ".join(ExtraDeclarations.get(header, []))
    if extra_declarations:
        extra_declarations = (
            " {"
            "  key: libcpp-header-exportable-declarations.ExtraDeclarations, "
            f' value: "{extra_declarations}" '
            "}, "
        )

    extra_header = ExtraHeader.get(header, "")
    if extra_header:
        extra_header = (
            "{"
            "  key: libcpp-header-exportable-declarations.ExtraHeader, "
            f' value: "{extra_header}" '
            "}, "
        )

    # Clang-tidy needs a file input
    print(f'// RUN{BLOCKLIT}: echo -e "' f"{include}" f'" > %t.{header}.cpp')
    print(
        f"// RUN{BLOCKLIT}: %{{clang-tidy}} %t.{header}.cpp "
        "  --checks='-*,libcpp-header-exportable-declarations' "
        "  -config='{CheckOptions: [ "
        f"   {{key: libcpp-header-exportable-declarations.Filename, value: {header}}}, "
        "    {key: libcpp-header-exportable-declarations.FileType, value: Header}, "
        f"   {skip_declarations} {extra_declarations} {extra_header}, "
        "  ]}' "
        "  --load=%{test-tools}/clang_tidy_checks/libcxx-tidy.plugin "
        "  -- %{flags} %{compile_flags} "
        f" | sort > %t.{header}.include"
    )

    # Compare the cppm and header file(s) return the same results.
    print(f"// RUN{BLOCKLIT}: diff -u %t.{header}.module %t.{header}.include")


# Merge the data of the parts
print(f"// RUN{BLOCKLIT}: sort -u -o %t.all_partitions %t.all_partitions")

# Dump the information as found in std.cppm.
print(
    f"// RUN{BLOCKLIT}: %{{clang-tidy}} %{{module}}/std.cppm "
    "  --checks='-*,libcpp-header-exportable-declarations' "
    "  -config='{CheckOptions: [ "
    "    {key: libcpp-header-exportable-declarations.Header, value: std.cppm}, "
    "    {key: libcpp-header-exportable-declarations.FileType, value: Module}, "
    "  ]}' "
    f" --load=%{{test-tools}}/clang_tidy_checks/libcxx-tidy.plugin "
    "  -- %{flags} %{compile_flags} "
    "  | sort > %t.module"
)


# Compare the sum of the parts with the main module.
print(f"// RUN{BLOCKLIT}: diff -u %t.all_partitions %t.module")

# Basic smoke test. Import a module and try to compile when using all
# exported names. This validates the clang-tidy script does not accidentally
# add named declarations to the list that are not available.
print(f"// RUN{BLOCKLIT}: echo 'import std;' > %t.compile.pass.cpp")
print(f"// RUN{BLOCKLIT}: cat %t.all_partitions >> %t.compile.pass.cpp")
print(f"// RUN{BLOCKLIT}: %{{cxx}} %{{flags}} %{{compile_flags}} -fsyntax-only %t.compile.pass.cpp")
