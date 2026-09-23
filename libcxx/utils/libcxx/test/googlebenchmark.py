# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

"""
Support for building GoogleBenchmark from the Lit configuration.

The benchmarks in the test suite are linked against GoogleBenchmark, which must be
built with the same Standard Library (and more generally with the same ABI-affecting
flags) as the benchmarks themselves. This allows building GoogleBenchmark on-demand
from Lit using the flags of the configuration being tested.

The result is cached inside the build directory so that subsequent invocations are
cheap.
"""

import hashlib
import os
import shlex
import subprocess

import lit.TestRunner

import libcxx.test.config
import libcxx.test.dsl

THIS_FILE = os.path.abspath(__file__)
LIBCXX_UTILS = os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE)))
MONOREPO_ROOT = os.path.dirname(os.path.dirname(LIBCXX_UTILS))
SOURCE_DIR = os.path.join(MONOREPO_ROOT, "third-party", "benchmark")

# Flags used by the test suite that must not be used when building GoogleBenchmark.
# Anything that isn't listed here is forwarded verbatim.
#
# -Werror
#     Avoid failing GoogleBenchmark's build due to warnings.
# -D_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#     Only relevant when testing libc++ itself.
# -fmodules, -fcxx-modules, -fmodules-cache-path=
#     Modules are irrelevant when building a third-party static library, and sharing a
#     module cache with the test suite is undesirable.
# -std=
#     GoogleBenchmark sets CMAKE_CXX_STANDARD itself and requires C++17.
_DROPPED_FLAGS = {
    "-Werror",
    "-D_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER",
    "-fmodules",
    "-fcxx-modules",
}

_DROPPED_FLAG_PREFIXES = (
    "-std=",
    "-fmodules-cache-path=",
)

# Flags that are dropped along with the '-Xclang' that introduces them.
_DROPPED_XCLANG_FLAGS = {
    "-fmodules-local-submodule-visibility",
}


def _expand(config, string):
    """
    Expand the Lit substitutions in the given string, recursively.
    """
    (expanded,) = lit.TestRunner.applySubstitutions(
        [string],
        config.substitutions,
        recursion_limit=config.recursiveExpansionLimit,
    )
    return expanded


def _filterFlags(flags):
    """
    Remove the flags that must not be used when building GoogleBenchmark.
    """
    result = []
    flags = iter(flags)
    for flag in flags:
        if flag == "-Xclang":
            arg = next(flags, None)
            if arg is None:
                result.append(flag)
            elif arg not in _DROPPED_XCLANG_FLAGS:
                result += [flag, arg]
        elif flag not in _DROPPED_FLAGS and not flag.startswith(_DROPPED_FLAG_PREFIXES):
            result.append(flag)
    return result


def _getFlags(config, substitutions):
    """
    Return the flags contained in the given substitutions, based on the flags used by the
    configuration under test.
    """
    flags = []
    for substitution in substitutions:
        expanded = _expand(config, _getSubstitution(substitution, config))
        flags += shlex.split(expanded)
    return _filterFlags(flags)


def _splitLibraries(flags):
    """
    Split the given link flags into (flags, libraries), where libraries contains the
    name of the libraries that were being linked against.

    We can't simply hand the libraries over to CMake as part of CMAKE_CXX_FLAGS, since
    CMake puts those flags before the object files on the link line.
    """
    result = []
    libraries = []
    flags = iter(flags)
    for flag in flags:
        if flag == "-l":
            library = next(flags, None)
            if library is None:
                result.append(flag)
            else:
                libraries.append(library)
        elif flag.startswith("-l"):
            libraries.append(flag[len("-l") :])
        else:
            result.append(flag)
    return (result, libraries)


def _getSubstitution(substitution, config):
    return libcxx.test.config._getSubstitution(substitution, config.substitutions)


def _fingerprint(config, flags, libraries):
    """
    Return an opaque value identifying this GoogleBenchmark build.

    This changes whenever the compiler is rebuilt or whenever the flags used to build
    GoogleBenchmark change.
    """
    compiler = libcxx.test.dsl._compilerFingerprint(config)
    return hashlib.sha256(repr((compiler, flags, libraries)).encode()).hexdigest()[:16]


def _run(litConfig, what, command, cwd):
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    if result.returncode != 0:
        pretty = " ".join(shlex.quote(arg) for arg in command)
        litConfig.fatal(
            "Failed to {} GoogleBenchmark.\n"
            "Command was:\n{}\n\n"
            "Output was:\n{}".format(what, pretty, result.stdout)
        )


def prepare(config, litConfig):
    """
    Make GoogleBenchmark available to the test suite and return the flags required to
    build the benchmarks against it.

    GoogleBenchmark is built using the same flags as the rest of the test suite, and the
    result is cached inside the build directory. The cache is keyed on the flags being
    used, so different Lit configurations do not interfere with each other.
    """
    flags, libraries = _splitLibraries(
        _getFlags(config, ("%{flags}", "%{compile_flags}", "%{link_flags}"))
    )
    root = os.path.join(config.test_exec_root, "__gbench__")
    prefix = os.path.join(root, _fingerprint(config, flags, libraries))
    buildDir = os.path.join(prefix, "build")
    installDir = os.path.join(prefix, "install")
    os.makedirs(root, exist_ok=True)

    cmake = os.environ.get("CMAKE", "cmake")

    if not os.path.exists(os.path.join(buildDir, "CMakeCache.txt")):
        litConfig.note("Configuring GoogleBenchmark in {}".format(buildDir))
        compiler = _expand(config, _getSubstitution("%{cxx}", config))
        _run(
            litConfig,
            "configure",
            [
                cmake,
                "-S", SOURCE_DIR,
                "-B", buildDir,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CXX_COMPILER={}".format(compiler),
                "-DCMAKE_CXX_FLAGS={}".format(" ".join(flags)),
                # Set CMAKE_EXE_LINKER_FLAGS in addition to BENCHMARK_CXX_LIBRARIES since we
                # need CMake's own probe executables to have the right linker flags.
                "-DCMAKE_EXE_LINKER_FLAGS={}".format(" ".join("-l{}".format(lib) for lib in libraries)),
                "-DCMAKE_INSTALL_PREFIX={}".format(installDir),
                "-DCMAKE_INSTALL_LIBDIR=lib",
                "-DBENCHMARK_CXX_LIBRARIES={}".format(";".join(libraries)),
                "-DBENCHMARK_ENABLE_TESTING=OFF",
                "-DBENCHMARK_ENABLE_WERROR=OFF",
                "-DBENCHMARK_INSTALL_DOCS=OFF",
            ],
            cwd=root,
        )

    # Always build: GoogleBenchmark is compiled against the headers of the library
    # under test, so it must be rebuilt when those change. This is a no-op when
    # nothing changed.
    _run(
        litConfig,
        "build",
        [cmake, "--build", buildDir, "--target", "install", "--parallel", str(os.cpu_count() or 1)],
        cwd=root,
    )

    include = os.path.join(installDir, "include")
    lib = os.path.join(installDir, "lib")
    return "-isystem {} -L {} -l benchmark".format(include, lib)
