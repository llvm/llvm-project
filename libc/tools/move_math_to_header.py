#!/usr/bin/env python3
"""
Migration script to move LLVM libc math functions from
libc/src/math/generic/<func>.cpp to header-only implementations in
libc/src/__support/math/<func>.h.

Usage:
    libc/tools/move_math_to_header.py <func_name>
"""

import argparse
import os
import re
import sys
from pathlib import Path


def find_llvm_root(start_path: Path) -> Path:
    """Find the LLVM project root by looking for .git directory."""
    current = start_path.resolve()
    while current != current.parent:
        if (current / ".git").exists() and (current / "libc").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not find LLVM project root. "
        "Please specify --root or run from within the llvm-project directory."
    )


def read_file(path: Path) -> str:
    """Read file content."""
    with open(path, "r") as f:
        return f.read()


def write_file(path: Path, content: str) -> None:
    """Write content to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def extract_llvm_libc_function(content: str) -> tuple:
    """
    Extract function signature from LLVM_LIBC_FUNCTION macro.
    Returns (return_type, func_name, params_with_types, param_names).
    """
    # Match LLVM_LIBC_FUNCTION(return_type, func_name, (params))
    pattern = r"LLVM_LIBC_FUNCTION\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*\(([^)]*)\)\s*\)"
    match = re.search(pattern, content)
    if not match:
        raise ValueError("Could not find LLVM_LIBC_FUNCTION macro")

    return_type = match.group(1)
    func_name = match.group(2)
    params_with_types = match.group(3).strip()

    # Extract just parameter names
    param_names = []
    for param in params_with_types.split(","):
        param = param.strip()
        if param:
            # Get the last word (parameter name)
            parts = param.split()
            if parts:
                param_names.append(parts[-1])

    return return_type, func_name, params_with_types, ", ".join(param_names)


def transform_cpp_to_header(content: str, func_name: str) -> str:
    """Transform the .cpp file content to a header-only implementation."""
    lines = content.splitlines()
    license_header = "\n".join(lines[:7])

    includes_end = lines.index("namespace LIBC_NAMESPACE_DECL {")
    include_lines = lines[7:includes_end]
    header_includes = []
    for inc in include_lines:
        # Skip public header
        if f'"src/math/{func_name}.h"' in inc:
            continue
        # Skip common.h (contains LLVM_LIBC_FUNCTION)
        if '"src/__support/common.h"' in inc:
            continue
        # Rely on clang-format to figure out include order.
        header_includes.append(inc.replace("src/__support/math/", ""))
    headers = "\n".join(header_includes)

    func_upper = func_name.upper()

    header = (
        license_header
        + f"""

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_{func_upper}_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_{func_upper}_H

{headers}

namespace LIBC_NAMESPACE_DECL {{

namespace math {{

// XXX put function implementation here

}} // namespace math

}} // namespace LIBC_NAMESPACE_DECL

"""
    )
    header += f"#endif // LLVM_LIBC_SRC___SUPPORT_MATH_{func_upper}_H\n"

    return header


def create_shared_header(func_name: str) -> str:
    """Create the shared header file content."""
    func_upper = func_name.upper()

    prefix = f"//===-- Shared {func_name} function "
    suffix = "-*- C++ -*-===//"
    dashes_needed = 80 - len(prefix) - len(suffix)
    dashes = "-" * dashes_needed

    return f"""{prefix}{dashes}{suffix}
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SHARED_MATH_{func_upper}_H
#define LLVM_LIBC_SHARED_MATH_{func_upper}_H

#include "shared/libc_common.h"
#include "src/__support/math/{func_name}.h"

namespace LIBC_NAMESPACE_DECL {{
namespace shared {{

using math::{func_name};

}} // namespace shared
}} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SHARED_MATH_{func_upper}_H
"""


def create_wrapper_cpp(
    content: str,
    func_name: str,
    return_type: str,
    params_with_types: str,
    param_names: str,
) -> str:
    """Create the simplified wrapper .cpp file."""
    license_header = "\n".join(content.splitlines()[:7])
    return (
        license_header
        + f"""

#include "src/math/{func_name}.h"
#include "src/__support/math/{func_name}.h"

namespace LIBC_NAMESPACE_DECL {{

LLVM_LIBC_FUNCTION({return_type}, {func_name}, ({params_with_types})) {{ return math::{func_name}({param_names}); }}

}} // namespace LIBC_NAMESPACE_DECL

// FIXME: Move this to the header:
"""
        + content
    )


def extract_cmake_depends(cmake_content: str, func_name: str) -> list:
    """Extract DEPENDS list from add_entrypoint_object for the given function."""
    # Find the add_entrypoint_object block for this function
    pattern = rf"add_entrypoint_object\s*\(\s*{re.escape(func_name)}\s+"
    match = re.search(pattern, cmake_content)
    if not match:
        return []

    # Find the DEPENDS section
    start = match.end()
    # Find the closing paren, accounting for nested parens
    paren_count = 1
    end = start
    for i, char in enumerate(cmake_content[start:], start):
        if char == "(":
            paren_count += 1
        elif char == ")":
            paren_count -= 1
            if paren_count == 0:
                end = i
                break

    block = cmake_content[match.start() : end + 1]

    # Extract DEPENDS entries
    depends_match = re.search(r"DEPENDS\s+(.*?)(?:\)|$)", block, re.DOTALL)
    if not depends_match:
        return []

    depends_text = depends_match.group(1)
    # Split by whitespace and filter
    deps = []
    for line in depends_text.split("\n"):
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue
        # Stop if we hit another keyword
        if line in ["SRCS", "HDRS", "COMPILE_OPTIONS"]:
            break
        # Skip libc.src.__support.common as it's only for LLVM_LIBC_FUNCTION,
        # which isn't needed in the new header.
        if "libc.src.__support.common" in line:
            continue
        deps.append(line)

    return deps


def update_generic_cmake(cmake_path: Path, func_name: str, extra_depends: list) -> str:
    """Update the generic CMakeLists.txt to simplify the entrypoint."""
    content = read_file(cmake_path)

    # Find the add_entrypoint_object block
    pattern = rf"(add_entrypoint_object\s*\(\s*{re.escape(func_name)}\s+)"
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Could not find add_entrypoint_object for {func_name}")

    # Find the full block
    start = match.start()
    paren_count = 0
    end = start
    found_open = False
    for i, char in enumerate(content[start:], start):
        if char == "(":
            paren_count += 1
            found_open = True
        elif char == ")":
            paren_count -= 1
            if found_open and paren_count == 0:
                end = i + 1
                break

    new_depends = [f"libc.src.__support.math.{func_name}"] + extra_depends
    deps_str = "\n    ".join(sorted(new_depends))

    # Create new simplified block
    new_block = f"""add_entrypoint_object(
  {func_name}
  SRCS
    {func_name}.cpp
  HDRS
    ../{func_name}.h
  DEPENDS
    {deps_str}
)"""

    return content[:start] + new_block + content[end:]


def transform_deps_for_math_cmake(deps: list) -> list:
    """Transform dependencies for the math CMakeLists.txt."""
    result = []
    for dep in deps:
        dep = dep.strip()
        if not dep:
            continue
        # Convert libc.src.__support.math.X to .X
        if dep.startswith("libc.src.__support.math."):
            short_name = dep.replace("libc.src.__support.math.", ".")
            result.append(short_name)
        else:
            result.append(dep)
    return sorted(result)


def update_support_math_cmake(cmake_path: Path, func_name: str, deps: list) -> str:
    """Add new add_header_library entry to support/math CMakeLists.txt."""
    content = read_file(cmake_path)

    transformed_deps = transform_deps_for_math_cmake(deps)

    # Build the new entry
    deps_str = "\n    ".join(transformed_deps) if transformed_deps else ""
    new_entry = f"""add_header_library(
  {func_name}
  HDRS
    {func_name}.h
  DEPENDS
    {deps_str}
)
"""

    # Find where to insert (after other add_header_library entries, alphabetically)
    insert_pos = 0
    for match in re.finditer(r"add_header_library\s*\(\s*(\w+)", content):
        existing_name = match.group(1)
        if existing_name < func_name:
            # Find end of this block
            block_start = match.start()
            paren_count = 0
            found_open = False
            for i, char in enumerate(content[block_start:], block_start):
                if char == "(":
                    paren_count += 1
                    found_open = True
                elif char == ")":
                    paren_count -= 1
                    if found_open and paren_count == 0:
                        insert_pos = i + 1
                        break
        elif existing_name > func_name:
            # Insert before this entry
            if insert_pos == 0:
                insert_pos = match.start()
            break

    if insert_pos > 0:
        # Find next newline after insert_pos
        next_newline = content.find("\n", insert_pos)
        if next_newline != -1:
            insert_pos = next_newline + 1
        # Insert the new entry
        return content[:insert_pos] + "\n" + new_entry + content[insert_pos:]

    # Fallback: append to end
    return content.rstrip() + "\n\n" + new_entry


def update_shared_math_h(path: Path, func_name: str) -> str:
    """Add include for new shared header to shared/math.h."""
    content = read_file(path)

    new_include = f'#include "math/{func_name}.h"'

    # Find the last #include "math/ line and insert after it
    lines = content.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('#include "math/'):
            insert_idx = i + 1

    # Insert alphabetically
    for i, line in enumerate(lines):
        if line.startswith('#include "math/'):
            # Extract the function name from the include
            match = re.search(r'#include "math/(\w+)\.h"', line)
            if match and match.group(1) > func_name:
                insert_idx = i
                break

    lines.insert(insert_idx, new_include)
    return "\n".join(lines)


def update_shared_test_cmake(path: Path, func_name: str) -> str:
    """Add dependency to shared test CMakeLists.txt."""
    content = read_file(path)

    new_dep = f"    libc.src.__support.math.{func_name}"

    # Find the DEPENDS section in add_fp_unittest
    # Insert before the closing paren of the DEPENDS block
    lines = content.split("\n")
    result = []
    in_depends = False

    for i, line in enumerate(lines):
        if "DEPENDS" in line:
            in_depends = True
        if in_depends and line.strip() == ")":
            # Check if dependency already exists
            if new_dep.strip() not in content:
                result.append(new_dep)
            in_depends = False
        result.append(line)

    return "\n".join(result)


def extract_bazel_deps(bazel_content: str, func_name: str) -> list:
    """Extract additional_deps from libc_math_function entry in BUILD.bazel."""
    # Find the libc_math_function entry
    pattern = rf'libc_math_function\s*\(\s*name\s*=\s*"{re.escape(func_name)}"'
    match = re.search(pattern, bazel_content)
    if not match:
        return []

    # Find the closing paren
    start = match.start()
    paren_count = 0
    end = start
    found_open = False
    for i, char in enumerate(bazel_content[start:], start):
        if char == "(":
            paren_count += 1
            found_open = True
        elif char == ")":
            paren_count -= 1
            if found_open and paren_count == 0:
                end = i + 1
                break

    block = bazel_content[start:end]

    # Extract additional_deps
    deps_match = re.search(r"additional_deps\s*=\s*\[(.*?)\]", block, re.DOTALL)
    if not deps_match:
        return []

    deps_text = deps_match.group(1)
    deps = []
    for match in re.finditer(r'"([^"]+)"', deps_text):
        deps.append(match.group(1))

    return deps


def update_bazel_build(bazel_path: Path, func_name: str) -> str:
    """Update BUILD.bazel with new support library and updated math function."""
    content = read_file(bazel_path)

    # Extract deps from existing libc_math_function
    deps = extract_bazel_deps(content, func_name)

    # Create new libc_support_library entry
    deps_str = '",\n        "'.join(deps) if deps else ""
    new_support_lib = f"""libc_support_library(
    name = "__support_math_{func_name}",
    hdrs = ["src/__support/math/{func_name}.h"],
    deps = [
        "{deps_str}",
    ],
)
"""

    # Find where to insert (after other __support_math_ entries, alphabetically)
    # Look for existing __support_math_ entries
    insert_pos = 0
    for match in re.finditer(
        r'libc_support_library\s*\(\s*name\s*=\s*"__support_math_(\w+)"', content
    ):
        existing_name = match.group(1)
        if existing_name < func_name:
            # Find end of this block
            block_start = match.start()
            paren_count = 0
            found_open = False
            for i, char in enumerate(content[block_start:], block_start):
                if char == "(":
                    paren_count += 1
                    found_open = True
                elif char == ")":
                    paren_count -= 1
                    if found_open and paren_count == 0:
                        insert_pos = i + 1
                        break
        elif existing_name > func_name:
            # Insert before this entry
            if insert_pos == 0:
                insert_pos = match.start()
            break

    if insert_pos > 0:
        # Find next newline after insert_pos
        next_newline = content.find("\n", insert_pos)
        if next_newline != -1:
            insert_pos = next_newline + 1

    # Insert the new support library
    content = content[:insert_pos] + "\n" + new_support_lib + content[insert_pos:]

    # Update the libc_math_function entry
    pattern = rf'(libc_math_function\s*\(\s*name\s*=\s*"{re.escape(func_name)}"\s*,\s*additional_deps\s*=\s*\[)[^\]]*(\]\s*,?\s*\))'
    replacement = rf'\1\n        ":__support_math_{func_name}",\n    \2'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    return content


def migrate_function(func_name: str, root: Path, dry_run: bool = False) -> None:
    """Perform the full migration of a math function."""

    src_cpp = root / "libc" / "src" / "math" / "generic" / f"{func_name}.cpp"
    if not src_cpp.exists():
        raise FileNotFoundError(f"Source file not found: {src_cpp}")

    support_header = root / "libc" / "src" / "__support" / "math" / f"{func_name}.h"
    if support_header.exists() and not dry_run:
        raise FileExistsError(f"Header already exists: {support_header}")

    print(f"Migrating {func_name}...")

    src_content = read_file(src_cpp)
    return_type, _, params_with_types, param_names = extract_llvm_libc_function(
        src_content
    )
    print(f"  Function signature: {return_type} {func_name}({params_with_types})")

    print("\nGenerating files...")

    generic_cmake = root / "libc" / "src" / "math" / "generic" / "CMakeLists.txt"
    cmake_content = read_file(generic_cmake)

    deps = extract_cmake_depends(cmake_content, func_name)
    # "libc.src.errno.errno" stays in generic_cmake and does not go not in support_cmake.
    extra_depends = []
    if "libc.src.errno.errno" in deps:
        extra_depends.append("libc.src.errno.errno")
        deps.remove("libc.src.errno.errno")

    updated_generic_cmake = update_generic_cmake(
        generic_cmake, func_name, extra_depends
    )
    print(f"  Updating: {generic_cmake}")
    if not dry_run:
        write_file(generic_cmake, updated_generic_cmake)

    support_cmake = root / "libc" / "src" / "__support" / "math" / "CMakeLists.txt"
    updated_support_cmake = update_support_math_cmake(support_cmake, func_name, deps)
    print(f"  Updating: {support_cmake}")
    if not dry_run:
        write_file(support_cmake, updated_support_cmake)

    test_cmake = root / "libc" / "test" / "shared" / "CMakeLists.txt"
    updated_test_cmake = update_shared_test_cmake(test_cmake, func_name)
    print(f"  Updating: {test_cmake}")
    if not dry_run:
        write_file(test_cmake, updated_test_cmake)

    bazel_build = (
        root / "utils" / "bazel" / "llvm-project-overlay" / "libc" / "BUILD.bazel"
    )
    updated_bazel = update_bazel_build(bazel_build, func_name)
    print(f"  Updating: {bazel_build}")
    if not dry_run:
        write_file(bazel_build, updated_bazel)

    shared_header = root / "libc" / "shared" / "math" / f"{func_name}.h"
    shared_content = create_shared_header(func_name)
    print(f"  Creating: {shared_header}")
    if not dry_run:
        write_file(shared_header, shared_content)

    shared_math_h = root / "libc" / "shared" / "math.h"
    updated_shared_math = update_shared_math_h(shared_math_h, func_name)
    print(f"  Updating: {shared_math_h}")
    if not dry_run:
        write_file(shared_math_h, updated_shared_math)

    # FIXME: More automation for the files below?

    header_content = transform_cpp_to_header(src_content, func_name)
    print(f"  Creating: {support_header}")
    if not dry_run:
        write_file(support_header, header_content)

    wrapper_content = create_wrapper_cpp(
        src_content, func_name, return_type, params_with_types, param_names
    )
    print(f"  Updating: {src_cpp}")
    if not dry_run:
        write_file(src_cpp, wrapper_content)

    test_cpp = root / "libc" / "test" / "shared" / "shared_math_test.cpp"

    print("\nMigration complete!")
    print("\nNext steps:")
    print(f"  1. git add {shared_header} {support_header}")
    print(f"  2. Move code from {src_cpp} to {support_header}")
    print(
        f"    * Replace `namespace {{` with `namespace {func_name}_internal` (including closing comment)"
    )
    print(f"    * Add `using namespace {func_name}_internal` to top of main function")
    print(f"    * Make all functions `LIBC_INLINE static`")
    print(f"    * Make all variables `LIBC_INLINE_VAR`")
    print(f"  3. Add a test to {test_cpp}")
    print(
        f"""  4. git commit -am '[libc][math] Refactor {func_name} implementation to header-only in src/__support/math folder.

Part of #147386

in preparation for:

https://discourse.llvm.org/t/rfc-make-clang-builtin-math-functions-constexpr-with-llvm-libc-to-support-c-23-constexpr-math-functions/86450'"""
    )
    print(f"  5. Run `git clang-format main` to fix include order")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate LLVM libc math function to header-only implementation"
    )
    parser.add_argument("func_name", help="Name of the math function to migrate")
    parser.add_argument(
        "--root",
        type=Path,
        help="Path to llvm-project root (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )

    args = parser.parse_args()

    if args.root:
        root = args.root.resolve()
    else:
        root = find_llvm_root(Path.cwd())

    print(f"LLVM project root: {root}")

    migrate_function(args.func_name, root, args.dry_run)


if __name__ == "__main__":
    main()
