# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Computes the list of projects that need to be tested from a diff.

Does some things, spits out a list of projects.
"""

from collections.abc import Set
import pathlib
import platform
import sys

PROJECT_DEPENDENCIES = {
    "llvm": set(),
    "clang": {"llvm"},
    "bolt": {"clang", "lld", "llvm"},
    "clang-tools-extra": {"clang", "llvm"},
    "compiler-rt": {"clang", "lld"},
    "libc": {"clang", "lld"},
    "openmp": {"clang", "lld"},
    "flang": {"llvm", "clang"},
    "lldb": {"llvm", "clang"},
    "libclc": {"llvm", "clang"},
    "lld": {"llvm"},
    "mlir": {"llvm"},
    "polly": {"llvm"},
}

DEPENDENTS_TO_TEST = {
    "llvm": {"bolt", "clang", "clang-tools-extra", "lld", "lldb", "mlir", "polly"},
    "lld": {"bolt", "cross-project-tests"},
    "clang": {"clang-tools-extra", "compiler-rt", "cross-project-tests"},
    "clang-tools-extra": {"libc"},
    "mlir": {"flang"},
}

DEPENDENT_RUNTIMES_TO_TEST = {"clang": {"libcxx", "libcxxabi", "libunwind"}}

EXCLUDE_LINUX = {
    "cross-project-tests",  # Tests are failing.
    "openmp",  # https://github.com/google/llvm-premerge-checks/issues/410
}

EXCLUDE_WINDOWS = {
    "cross-project-tests",  # Tests are failing.
    "compiler-rt",  # Tests are taking too long.
    "openmp",  # Does not detect perl installation.
    "libc",  # No Windows Support.
    "lldb",  # Custom environment requirements.
    "bolt",  # No Windows Support.
}

EXCLUDE_MAC = {
    "bolt",
    "compiler-rt",
    "cross-project-tests",
    "flang",
    "libc",
    "libcxx",
    "libcxxabi",
    "libunwind",
    "lldb",
    "openmp",
    "polly",
}

PROJECT_CHECK_TARGETS = {
    "clang-tools-extra": "check-clang-tools",
    "compiler-rt": "check-compiler-rt",
    "cross-project-tests": "check-cross-project",
    "libcxx": "check-cxx",
    "libcxxabi": "check-cxxabi",
    "libunwind": "check-unwind",
    "lldb": "check-lldb",
    "llvm": "check-llvm",
    "clang": "check-clang",
    "bolt": "check-bolt",
    "lld": "check-lld",
    "flang": "check-flang",
    "libc": "check-libc",
    "lld": "check-lld",
    "lldb": "check-lldb",
    "mlir": "check-mlir",
    "openmp": "check-openmp",
    "polly": "check-polly",
}


def _add_dependencies(projects: Set[str]) -> Set[str]:
    projects_with_dependents = set(projects)
    current_projects_count = 0
    while current_projects_count != len(projects_with_dependents):
        current_projects_count = len(projects_with_dependents)
        for project in list(projects_with_dependents):
            if project not in PROJECT_DEPENDENCIES:
                continue
            projects_with_dependents.update(PROJECT_DEPENDENCIES[project])
    return projects_with_dependents


def _compute_projects_to_test(modified_projects: Set[str], platform: str) -> Set[str]:
    projects_to_test = set()
    for modified_project in modified_projects:
        # Skip all projects where we cannot run tests.
        if modified_project not in PROJECT_CHECK_TARGETS:
            continue
        projects_to_test.add(modified_project)
        if modified_project not in DEPENDENTS_TO_TEST:
            continue
        projects_to_test.update(DEPENDENTS_TO_TEST[modified_project])
    if platform == "Linux":
        for to_exclude in EXCLUDE_LINUX:
            if to_exclude in projects_to_test:
                projects_to_test.remove(to_exclude)
    elif platform == "Windows":
        for to_exclude in EXCLUDE_WINDOWS:
            if to_exclude in projects_to_test:
                projects_to_test.remove(to_exclude)
    elif platform == "Darwin":
        for to_exclude in EXCLUDE_MAC:
            if to_exclude in projects_to_test:
                projects_to_test.remove(to_exclude)
    else:
        raise ValueError("Unexpected platform.")
    return projects_to_test


def _compute_projects_to_build(projects_to_test: Set[str]) -> Set[str]:
    return _add_dependencies(projects_to_test)


def _compute_project_check_targets(projects_to_test: Set[str]) -> Set[str]:
    check_targets = set()
    for project_to_test in projects_to_test:
        if project_to_test not in PROJECT_CHECK_TARGETS:
            continue
        check_targets.add(PROJECT_CHECK_TARGETS[project_to_test])
    return check_targets


def _compute_runtimes_to_test(projects_to_test: Set[str]) -> Set[str]:
    runtimes_to_test = set()
    for project_to_test in projects_to_test:
        if project_to_test not in DEPENDENT_RUNTIMES_TO_TEST:
            continue
        runtimes_to_test.update(DEPENDENT_RUNTIMES_TO_TEST[project_to_test])
    return runtimes_to_test


def _compute_runtime_check_targets(runtimes_to_test: Set[str]) -> Set[str]:
    check_targets = set()
    for runtime_to_test in runtimes_to_test:
        check_targets.add(PROJECT_CHECK_TARGETS[runtime_to_test])
    return check_targets


def _get_modified_projects(modified_files: list[str]) -> Set[str]:
    modified_projects = set()
    for modified_file in modified_files:
        modified_projects.add(pathlib.Path(modified_file).parts[0])
    return modified_projects


def get_env_variables(modified_files: list[str], platform: str) -> Set[str]:
    modified_projects = _get_modified_projects(modified_files)
    projects_to_test = _compute_projects_to_test(modified_projects, platform)
    projects_to_build = _compute_projects_to_build(projects_to_test)
    projects_check_targets = _compute_project_check_targets(projects_to_test)
    runtimes_to_test = _compute_runtimes_to_test(projects_to_test)
    runtimes_check_targets = _compute_runtime_check_targets(runtimes_to_test)
    return {
        "projects_to_build": ";".join(sorted(projects_to_build)),
        "project_check_targets": " ".join(sorted(projects_check_targets)),
        "runtimes_to_build": ";".join(sorted(runtimes_to_test)),
        "runtimes_check_targets": " ".join(sorted(runtimes_check_targets)),
    }


if __name__ == "__main__":
    current_platform = platform.system()
    if len(sys.argv) == 2:
        current_platform = sys.argv[1]
    env_variables = get_env_variables(sys.stdin.readlines(), current_platform)
    for env_variable in env_variables:
        print(f"{env_variable}=\"{env_variables[env_variable]}\"")
