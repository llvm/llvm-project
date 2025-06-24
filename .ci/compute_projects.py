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

# This mapping lists out the dependencies for each project. These should be
# direct dependencies. The code will handle transitive dependencies. Some
# projects might have optional dependencies depending upon how they are built.
# The dependencies listed here should be the dependencies required for the
# configuration built/tested in the premerge CI.
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

# This mapping describes the additional projects that should be tested when a
# specific project is touched. We enumerate them specifically rather than
# just invert the dependencies list to give more control over what exactly is
# tested.
DEPENDENTS_TO_TEST = {
    "llvm": {
        "bolt",
        "clang",
        "clang-tools-extra",
        "lld",
        "lldb",
        "mlir",
        "polly",
        "flang",
    },
    "lld": {"bolt", "cross-project-tests"},
    # TODO(issues/132795): LLDB should be enabled on clang changes.
    "clang": {"clang-tools-extra", "cross-project-tests"},
    "mlir": {"flang"},
    # Test everything if ci scripts are changed.
    ".ci": {
        "llvm",
        "clang",
        "lld",
        "lldb",
        "bolt",
        "clang-tools-extra",
        "mlir",
        "polly",
        "flang",
        "libclc",
        "openmp",
    },
}

# This mapping describes runtimes that should be enabled for a specific project,
# but not necessarily run for testing. The only case of this currently is lldb
# which needs some runtimes enabled for tests.
DEPENDENT_RUNTIMES_TO_BUILD = {"lldb": {"libcxx", "libcxxabi", "libunwind"}}

# This mapping describes runtimes that should be tested when the key project is
# touched.
DEPENDENT_RUNTIMES_TO_TEST = {
    "clang": {"compiler-rt"},
    "clang-tools-extra": {"libc"},
    ".ci": {"compiler-rt", "libc"},
}
DEPENDENT_RUNTIMES_TO_TEST_NEEDS_RECONFIG = {
    "llvm": {"libcxx", "libcxxabi", "libunwind"},
    "clang": {"libcxx", "libcxxabi", "libunwind"},
    ".ci": {"libcxx", "libcxxabi", "libunwind"},
}

EXCLUDE_LINUX = {
    "cross-project-tests",  # TODO(issues/132796): Tests are failing.
    "openmp",  # https://github.com/google/llvm-premerge-checks/issues/410
}

EXCLUDE_WINDOWS = {
    "cross-project-tests",  # TODO(issues/132797): Tests are failing.
    "compiler-rt",  # TODO(issues/132798): Tests take excessive time.
    "openmp",  # TODO(issues/132799): Does not detect perl installation.
    "libc",  # No Windows Support.
    "lldb",  # TODO(issues/132800): Needs environment setup.
    "bolt",  # No Windows Support.
}

# These are projects that we should test if the project itself is changed but
# where testing is not yet stable enough for it to be enabled on changes to
# dependencies.
EXCLUDE_DEPENDENTS_WINDOWS = {
    "flang",  # TODO(issues/132803): Flang is not stable.
}

EXCLUDE_MAC = {
    "bolt",
    "compiler-rt",
    "cross-project-tests",
    "flang",
    "libc",
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

RUNTIMES = {"libcxx", "libcxxabi", "libunwind", "compiler-rt", "libc"}


def _add_dependencies(projects: Set[str], runtimes: Set[str]) -> Set[str]:
    projects_with_dependents = set(projects)
    current_projects_count = 0
    while current_projects_count != len(projects_with_dependents):
        current_projects_count = len(projects_with_dependents)
        for project in list(projects_with_dependents):
            if project in PROJECT_DEPENDENCIES:
                projects_with_dependents.update(PROJECT_DEPENDENCIES[project])
    for runtime in runtimes:
        if runtime in PROJECT_DEPENDENCIES:
            projects_with_dependents.update(PROJECT_DEPENDENCIES[runtime])
    return projects_with_dependents


def _exclude_projects(current_projects: Set[str], platform: str) -> Set[str]:
    if platform == "Linux":
        to_exclude = EXCLUDE_LINUX
    elif platform == "Windows":
        to_exclude = EXCLUDE_WINDOWS
    elif platform == "Darwin":
        to_exclude = EXCLUDE_MAC
    else:
        raise ValueError(f"Unexpected platform: {platform}")
    return current_projects.difference(to_exclude)


def _compute_projects_to_test(modified_projects: Set[str], platform: str) -> Set[str]:
    projects_to_test = set()
    for modified_project in modified_projects:
        if modified_project in RUNTIMES:
            continue
        # Skip all projects where we cannot run tests.
        if modified_project in PROJECT_CHECK_TARGETS:
            projects_to_test.add(modified_project)
        if modified_project not in DEPENDENTS_TO_TEST:
            continue
        for dependent_project in DEPENDENTS_TO_TEST[modified_project]:
            if (
                platform == "Windows"
                and dependent_project in EXCLUDE_DEPENDENTS_WINDOWS
            ):
                continue
            projects_to_test.add(dependent_project)
    projects_to_test = _exclude_projects(projects_to_test, platform)
    return projects_to_test


def _compute_projects_to_build(
    projects_to_test: Set[str], runtimes: Set[str]
) -> Set[str]:
    return _add_dependencies(projects_to_test, runtimes)


def _compute_project_check_targets(projects_to_test: Set[str]) -> Set[str]:
    check_targets = set()
    for project_to_test in projects_to_test:
        if project_to_test in PROJECT_CHECK_TARGETS:
            check_targets.add(PROJECT_CHECK_TARGETS[project_to_test])
    return check_targets


def _compute_runtimes_to_test(modified_projects: Set[str], platform: str) -> Set[str]:
    runtimes_to_test = set()
    for modified_project in modified_projects:
        if modified_project in DEPENDENT_RUNTIMES_TO_TEST:
            runtimes_to_test.update(DEPENDENT_RUNTIMES_TO_TEST[modified_project])
    return _exclude_projects(runtimes_to_test, platform)


def _compute_runtimes_to_test_needs_reconfig(
    modified_projects: Set[str], platform: str
) -> Set[str]:
    runtimes_to_test = set()
    for modified_project in modified_projects:
        if modified_project in DEPENDENT_RUNTIMES_TO_TEST_NEEDS_RECONFIG:
            runtimes_to_test.update(
                DEPENDENT_RUNTIMES_TO_TEST_NEEDS_RECONFIG[modified_project]
            )
    return _exclude_projects(runtimes_to_test, platform)


def _compute_runtimes_to_build(
    runtimes_to_test: Set[str], modified_projects: Set[str], platform: str
) -> Set[str]:
    runtimes_to_build = set(runtimes_to_test)
    for modified_project in modified_projects:
        if modified_project in DEPENDENT_RUNTIMES_TO_BUILD:
            runtimes_to_build.update(DEPENDENT_RUNTIMES_TO_BUILD[modified_project])
    return _exclude_projects(runtimes_to_build, platform)


def _get_modified_projects(modified_files: list[str]) -> Set[str]:
    modified_projects = set()
    for modified_file in modified_files:
        path_parts = pathlib.Path(modified_file).parts
        # Exclude files in the docs directory. They do not impact an test
        # targets and there is a separate workflow used for ensuring the
        # documentation builds.
        if len(path_parts) > 2 and path_parts[1] == "docs":
            continue
        # Exclude files for the gn build. We do not test it within premerge
        # and changes occur often enough that they otherwise take up
        # capacity.
        if len(path_parts) > 3 and path_parts[:3] == ("llvm", "utils", "gn"):
            continue
        modified_projects.add(pathlib.Path(modified_file).parts[0])
    return modified_projects


def get_env_variables(modified_files: list[str], platform: str) -> Set[str]:
    modified_projects = _get_modified_projects(modified_files)
    projects_to_test = _compute_projects_to_test(modified_projects, platform)
    runtimes_to_test = _compute_runtimes_to_test(modified_projects, platform)
    runtimes_to_test_needs_reconfig = _compute_runtimes_to_test_needs_reconfig(
        modified_projects, platform
    )
    runtimes_to_build = _compute_runtimes_to_build(
        runtimes_to_test | runtimes_to_test_needs_reconfig, modified_projects, platform
    )
    projects_to_build = _compute_projects_to_build(projects_to_test, runtimes_to_build)
    projects_check_targets = _compute_project_check_targets(projects_to_test)
    runtimes_check_targets = _compute_project_check_targets(runtimes_to_test)
    runtimes_check_targets_needs_reconfig = _compute_project_check_targets(
        runtimes_to_test_needs_reconfig
    )
    # We use a semicolon to separate the projects/runtimes as they get passed
    # to the CMake invocation and thus we need to use the CMake list separator
    # (;). We use spaces to separate the check targets as they end up getting
    # passed to ninja.
    return {
        "projects_to_build": ";".join(sorted(projects_to_build)),
        "project_check_targets": " ".join(sorted(projects_check_targets)),
        "runtimes_to_build": ";".join(sorted(runtimes_to_build)),
        "runtimes_check_targets": " ".join(sorted(runtimes_check_targets)),
        "runtimes_check_targets_needs_reconfig": " ".join(
            sorted(runtimes_check_targets_needs_reconfig)
        ),
    }


if __name__ == "__main__":
    current_platform = platform.system()
    if len(sys.argv) == 2:
        current_platform = sys.argv[1]
    env_variables = get_env_variables(sys.stdin.readlines(), current_platform)
    for env_variable in env_variables:
        print(f"{env_variable}='{env_variables[env_variable]}'")
