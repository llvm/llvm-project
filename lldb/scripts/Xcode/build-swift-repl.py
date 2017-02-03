#! /usr/bin/env python

# build-swift-repl.py
#
# The Swift REPL must be built using the newly-built
# Swift compiler or we can't debug it.

import os
import re
import shutil
import sys

import lldbbuild

# Command-line interface


def check_args():
    if len(sys.argv) != 3:
        print "usage: " + sys.argv[0] + " REPL_EXECUTABLE REPL_SOURCE_FILE"
        sys.exit(1)


def repl_executable():
    return sys.argv[1]


def repl_source_file():
    return sys.argv[2]

# Xcode interface


def lldb_build_path():
    return os.environ.get('CONFIGURATION_BUILD_DIR')


def arch():
    return os.environ.get("ARCHS").split()[0]


def macosx_deployment_target():
    return os.environ.get("MACOSX_DEPLOYMENT_TARGET")


def repl_rpaths():
    return os.environ.get("REPL_SWIFT_RPATH").split()

# Arguments to swiftc


def module_cache_path():
    return os.path.join(lldb_build_path(), "repl_swift_module_cache")


def swiftc_path():
    return os.path.join(
        lldbbuild.expected_package_build_path_for("swift"),
        "bin",
        "swiftc")


def swift_target():
    deployment_target = macosx_deployment_target()
    if deployment_target is not None:
        return arch() + "-apple-macosx" + deployment_target
    else:
        return None


def target_arg_for_repl():
    target = swift_target()
    if target is not None:
        return ["-target", target]
    else:
        return []


def linker_args_for_rpath(rpath):
    return ["-Xlinker", "-rpath", "-Xlinker", rpath]


def linker_args_for_repl():
    return [
        arg for args in map(
            linker_args_for_rpath,
            repl_rpaths()) for arg in args]


def do_maybe_recursive_delete(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    assert (not os.path.exists(path))


def do_create_folder(path):
    os.makedirs(path)


def module_cache_args_for_repl():
    cache = module_cache_path()
    do_maybe_recursive_delete(cache)
    do_create_folder(cache)
    return ["-module-cache-path", cache]


def swiftc_args_for_repl():
    return [swiftc_path(),
            "-DXCODE_BUILD_ME"] + target_arg_for_repl() + ["-g",
                                                           "-o",
                                                           repl_executable(),
                                                           repl_source_file()] + module_cache_args_for_repl() + linker_args_for_repl()


def strip_args_for_repl():
    return ["strip", "-S", repl_executable()]

# Core logic

check_args()
lldbbuild.run_in_directory(swiftc_args_for_repl(), lldb_build_path())
lldbbuild.run_in_directory(strip_args_for_repl(), lldb_build_path())

sys.exit(0)
