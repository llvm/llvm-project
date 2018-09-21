#!/usr/bin/env python

# swift.py
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

import os
import os.path
import subprocess

configurations = ['Ninja-DebugAssert', 'Ninja-ReleaseAssert',
                  'Ninja-RelWithDebInfoAssert']  # TODO: add more configurations


def getSwiftCompiler():
    """Returns the swift compiler in effect the test suite is running with."""
    env_var = "SWIFTC"
    if env_var in os.environ:
        return os.environ[env_var]
    else:
        lldb_root_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "..")
        lldb_parent_path = os.path.join(lldb_root_path, "..")

        candidates = []

        candidates.append(
            os.path.join(
                lldb_root_path,
                "llvm-build/DebugPresubmission/swift-macosx-x86_64/bin/swiftc"))
        candidates.append(
            os.path.join(
                lldb_root_path,
                "PrebuiltSwiftlang/swift-macosx-x86_64/bin/swiftc"))
        candidates.append(
            os.path.join(
                lldb_root_path,
                "llvm-build/ReleaseAssert/swift-macosx-x86_64/bin/swiftc"))
        candidates.append(
            os.path.join(
                lldb_root_path,
                "llvm-build/DebugAssert/swift-macosx-x86_64/bin/swiftc"))
        candidates.append(
            os.path.join(
                lldb_root_path,
                "llvm-build/Debug/swift-macosx-x86_64/bin/swiftc"))

        for configuration in configurations:
            candidates.append(
                os.path.join(
                    lldb_parent_path,
                    'build',
                    configuration,
                    'swift-linux-x86_64/bin/swiftc'))
            candidates.append(
                os.path.join(
                    lldb_parent_path,
                    'build',
                    configuration,
                    'swift-macosx-x86_64/bin/swiftc'))

        for candidate in candidates:
            if os.path.exists(candidate):
                os.environ[env_var] = os.path.join(
                    os.path.realpath(
                        os.path.dirname(candidate)),
                    os.path.basename(candidate))
                return os.environ[env_var]

        # Give up and just return "swiftc"...
        return "swiftc"

swift_sdk_root = None


def getSwiftSDKRoot():
    """Returns the SDK root to be used for compiling Swift/ObjC interop code."""
    global swift_sdk_root
    if swift_sdk_root is None:
        import platform
        if "SDKROOT" in os.environ:
            swift_sdk_root = os.environ["SDKROOT"]
        if "SWIFTSDKROOT" in os.environ:
            swift_sdk_root = os.environ["SWIFTSDKROOT"]
        elif platform.system() == 'Darwin':
            try:
                sdk_path = subprocess.check_output(
                    'xcrun -sdk macosx --show-sdk-path', shell=True)
                if sdk_path[-1] == '\n':
                    sdk_path = sdk_path[0:-1]
                if os.path.isdir(sdk_path):
                    swift_sdk_root = sdk_path
            except:
                pass
    if swift_sdk_root is None:
        swift_sdk_root = "/"
    return swift_sdk_root


def getSwiftLibraryPath():
    """Returns the swift library include path for the linker for the swift the test suite is running with."""
    env_var = "SWIFTLIBS"
    if env_var in os.environ:
        return os.environ[env_var]
    else:
        lldb_root_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "..")
        lldb_parent_path = os.path.join(lldb_root_path, "..")

        candidates = []

        candidates.append(
            os.path.join(
                lldb_root_path,
                "llvm-build/DebugPresubmission/swift-macosx-x86_64/lib/swift"))
        candidates.append(
            os.path.join(
                lldb_root_path,
                "PrebuiltSwiftlang/swift-macosx-x86_64/lib/swiftc"))
        candidates.append(
            os.path.join(
                lldb_root_path,
                "llvm-build/ReleaseAssert/swift-macosx-x86_64/lib/swiftc"))
        candidates.append(
            os.path.join(
                lldb_root_path,
                "llvm-build/DebugAssert/swift-macosx-x86_64/lib/swiftc"))
        candidates.append(
            os.path.join(
                lldb_root_path,
                "llvm-build/Debug/swift-macosx-x86_64/lib/swiftc"))

        for configuration in configurations:
            candidates.append(
                os.path.join(
                    lldb_parent_path,
                    'build',
                    configuration,
                    'swift-linux-x86_64/lib/swift'))
            candidates.append(
                os.path.join(
                    lldb_parent_path,
                    'build',
                    configuration,
                    'swift-macosx-x86_64/lib/swift'))

        for candidate in candidates:
            if os.path.exists(candidate):
                os.environ[env_var] = os.path.join(
                    os.path.realpath(
                        os.path.dirname(candidate)),
                    os.path.basename(candidate))
                return os.environ[env_var]

        # Give up and just return "/usr/lib/swift"...
        return "/usr/lib/swift"
