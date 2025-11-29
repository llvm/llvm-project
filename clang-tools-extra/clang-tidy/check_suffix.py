#!/usr/bin/env python3
#
# ===-----------------------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#

import os
import sys

# This script verifies that all clang-tidy checks' header/source filenames end
# with the "Check" suffix.

FOLDERS = [
    "abseil",
    "altera",
    "android",
    "boost",
    "bugprone",
    "cert",
    "concurrency",
    "cppcoreguidelines",
    "custom",
    "darwin",
    "fuchsia",
    "google",
    "hicpp",
    "linuxkernel",
    "llvm",
    "llvmlibc",
    "misc",
    "modernize",
    "mpi",
    "objc",
    "openmp",
    "performance",
    "portability",
    "readability",
    "zircon",
]

GLOBAL_IGNORES = {
    "CMakeLists.txt",
}

# Per-folder ignores. Paths listed here are matched by basename; entries that
# refer to directories should be skipped from traversal.
PER_FOLDER_IGNORES = {
    "abseil": {"AbseilMatcher.h", "DurationRewriter.h", "DurationRewriter.cpp"},
    "cert": {"LICENSE.TXT"},
    "hicpp": {"LICENSE.TXT"},
    "llvmlibc": {"NamespaceConstants.h"},
    "misc": {"ConfusableTable"},
    "modernize": {
        "IntegralLiteralExpressionMatcher.h",
        "IntegralLiteralExpressionMatcher.cpp",
        "LoopConvertUtils.h",
        "LoopConvertUtils.cpp",
    },
}


def get_module_name(folder: str) -> str:
    """Return the tidy module filename for a folder."""
    special_cases = {
        "hicpp": "HICPPTidyModule.cpp",
        "cppcoreguidelines": "CppCoreGuidelinesTidyModule.cpp",
        "linuxkernel": "LinuxKernelTidyModule.cpp",
        "llvmlibc": "LLVMLibcTidyModule.cpp",
        "mpi": "MPITidyModule.cpp",
        "objc": "ObjCTidyModule.cpp",
        "openmp": "OpenMPTidyModule.cpp",
        "llvm": "LLVMTidyModule.cpp",
        "cert": "CERTTidyModule.cpp",
    }
    return special_cases.get(folder, f"{folder.capitalize()}TidyModule.cpp")


def should_skip(folder: str, path_basename: str) -> bool:
    """Whether a file/dir should be ignored for a folder."""
    if path_basename in GLOBAL_IGNORES:
        return True
    ignores = PER_FOLDER_IGNORES.get(folder, set())
    return path_basename in ignores


def iter_candidate_files(clang_tidy_path: str):
    """Yield all .h/.cpp files to check."""
    for folder in FOLDERS:
        folder_path = os.path.join(clang_tidy_path, folder)
        if not os.path.isdir(folder_path):
            continue

        module_file = get_module_name(folder)

        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if not should_skip(folder, d)]

            for name in files:
                if name == module_file or should_skip(folder, name):
                    continue
                if not (name.endswith(".h") or name.endswith(".cpp")):
                    continue
                rel = os.path.relpath(os.path.join(root, name), clang_tidy_path)
                yield rel


def find_missing_suffix_files(clang_tidy_path: str) -> list:
    """Return a list of .h/.cpp files for checks missing the 'Check' suffix."""
    missing: list = []
    for rel in iter_candidate_files(clang_tidy_path):
        base = os.path.splitext(os.path.basename(rel))[0]
        if not base.endswith("Check"):
            missing.append(rel)
    return missing


def main() -> None:
    clang_tidy_path = os.path.dirname(os.path.abspath(__file__))
    missing_files = find_missing_suffix_files(clang_tidy_path)

    if missing_files:
        print("Found checks missing the 'Check' suffix:")
        for file_path in sorted(missing_files):
            print(file_path)
    else:
        print("No checks found missing the 'Check' suffix.")

    if missing_files:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
