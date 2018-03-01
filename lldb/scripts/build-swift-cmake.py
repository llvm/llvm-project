#!/usr/bin/python

# build-swift-cmake.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------

import argparse
import fnmatch
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--lldb-extra-cmake-args', action='store',
                    help='extra arguments to be passed to lldb cmake')
parser.add_argument('--lldb-extra-xcodebuild-args', action='store',
                    help='extra arguments to be passed to lldb xcodebuild')
parser.add_argument('--update', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--curses', action='store_true',
                    help='test with curses test runners where available')
parser.add_argument('--release', action='store_true',
                    help='build in release mode')
parser.add_argument('--no-debugserver', action='store_true',
                    help='build without debugserver')
parser.add_argument(
    '--no-system-debugserver',
    action='store_false',
    dest='use_system_debugserver',
    help='do not copy in the system debugserver (default is to copy it in)')
parser.add_argument('--package', action='store_true',
                    help='build for packaging')
parser.add_argument('--foundation', action='store_true',
                    help='build swift foundation')

args = parser.parse_args()

def apply_patches(repo):
    patches_dir = os.path.dirname(os.path.realpath(__file__))   # patch files will be in scripts/ dir just like this script
    files = os.listdir(patches_dir)
    patches = [
        f for f in files if fnmatch.fnmatch(
            f, repo + '.*.diff')]
    for p in patches:
        subprocess.call(["patch", '-p1', '-i',
                         os.path.join(patches_dir, p)],
                         cwd=repo)

def checkout_git(dir, repo, branch):
    if not os.path.isdir(dir):
        subprocess.call(["git", "clone", "-b", branch, repo, dir])
        apply_patches(dir)


def update_git(dir):
    if os.path.isdir(dir):
        subprocess.call(["git", "fetch", "--all"], cwd=dir)
        subprocess.call(["git", "merge", "--ff-only", "@{upstream}"], cwd=dir)


def use_gold_linker():
    """@return True if the gold linker should be used; False otherwise."""
    return os.path.isfile("/usr/bin/ld.gold")

uname = sys.platform

checkout_git(
    "llvm",
    "https://github.com/apple/swift-llvm.git",
    "stable")
checkout_git(
    "clang",
    "https://github.com/apple/swift-clang.git",
    "stable")
checkout_git("swift", "https://github.com/apple/swift.git", "master")
checkout_git("cmark", "https://github.com/apple/swift-cmark.git", "master")
checkout_git("ninja", "https://github.com/ninja-build/ninja.git", "master")
checkout_git(
    "lldb",
    "https://github.com/apple/swift-lldb.git",
    "master")

if args.package:
    checkout_git(
        "llbuild",
        "https://github.com/apple/swift-llbuild.git",
        "master")
    checkout_git(
        "swiftpm",
        "https://github.com/apple/swift-package-manager.git",
        "master")
    checkout_git(
        "swift-corelibs-foundation",
        "https://github.com/apple/swift-corelibs-foundation.git",
        "master")
    checkout_git(
        "swift-corelibs-xctest",
        "https://github.com/apple/swift-corelibs-xctest.git",
        "master")
    checkout_git(
        "swift-integration-tests",
        "https://github.com/apple/swift-integration-tests.git",
        "master")
elif args.foundation:
    checkout_git(
        "swift-corelibs-foundation",
        "https://github.com/apple/swift-corelibs-foundation.git",
        "master")

if args.update:
    update_git("llvm")
    update_git("clang")
    update_git("swift")
    update_git("cmark")
    update_git("ninja")
    update_git("lldb")
    if args.package:
        update_git("llbuild")
        update_git("swiftpm")
        update_git("swift-corelibs-foundation")
        update_git("swift-corelibs-xctest")
        update_git("swift-integration-tests")
    elif args.foundation:
        update_git("swift-corelibs-foundation")

if not os.path.exists("install"):
    os.makedirs("install")

package_darwin = args.package and (uname == "Darwin")

build_script_arguments = []
build_script_impl_arguments = []

if args.lldb_extra_xcodebuild_args:
    build_script_impl_arguments.append(
        "--lldb-extra-xcodebuild-args={}".format(
            args.lldb_extra_xcodebuild_args))

if package_darwin:
    # packaging preset
    build_script_arguments += ["--preset=buildbot_osx_package"]

    if not os.path.exists("symroot"):
        os.makedirs("symroot")
    if not os.path.exists("package"):
        os.makedirs("package")
    build_script_arguments += [
        "install_destdir=" + os.getcwd() + "/install",
        "installable_package=" + os.getcwd() + "/package/package.tar.gz",
        "install_toolchain_dir=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain",
        "install_symroot=" + os.getcwd() + "/symroot",
        "symbols_package=" + os.getcwd() + "/package/symbols.tar.gz"]
elif args.package:
    print("--package is unsupported on non-OS X platforms")
else:
    if args.release:
        build_script_arguments += ["--release", "--assertions", "--lldb"]
    else:
        build_script_arguments += ["--debug-swift",
                                   "--debug-lldb", "--skip-build-benchmarks",
                                   "--extra-cmake-options=-DCMAKE_CXX_FLAGS=-fno-limit-debug-info"]
    if args.foundation:
        build_script_arguments += ["--foundation"]
    build_script_impl_arguments += ["--build-swift-static-stdlib=1"]

    if args.lldb_extra_cmake_args and len(args.lldb_extra_cmake_args) > 0:
        # Add the quoted version of the command line arg.
        build_script_impl_arguments.append(
            "--lldb-extra-cmake-args={}".format(args.lldb_extra_cmake_args))

    if uname != "Darwin":
        # we don't build with Xcode, so we can actually install
        # build_script_impl_arguments += [ "--install-swift", "--install-lldb", "--install-prefix", "/usr", "--install-destdir", os.getcwd() + "/install", "--swift-install-components=compiler;clang-builtin-headers;stdlib;stdlib-experimental;sdk-overlay;editor-integration;tools;testsuite-tools;dev" ]
        build_script_impl_arguments += [
            "--install-swift",
            "--install-lldb",
            "--install-destdir",
            os.path.join(os.getcwd(), "install"),
            "--swift-install-components=compiler;clang-builtin-headers;stdlib;stdlib-experimental;sdk-overlay;editor-integration;tools;testsuite-tools;dev"]

    # build_script_impl_arguments += ["--reconfigure"]

    # If we're on Linux, and if the /usr/bin/ld.gold exists, indicate we
    # want to use the gold linker.
    if use_gold_linker():
        build_script_impl_arguments.append("--use-gold-linker")

    if args.test:
        build_script_arguments += ["--test"]
        build_script_impl_arguments += ["--skip-test-cmark",
                                        "--skip-test-swift"]
        if args.curses:
            build_script_impl_arguments += ["--lldb-test-with-curses"]

    if args.no_debugserver:
        build_script_impl_arguments += ['--lldb-no-debugserver']
    elif args.use_system_debugserver:
        build_script_impl_arguments += ['--lldb-use-system-debugserver']

args = ["python", os.path.join("swift", "utils", "build-script")] + \
    build_script_arguments + ["--"] + build_script_impl_arguments

print(" ".join(args))

return_code = subprocess.call(args)
sys.exit(return_code)
