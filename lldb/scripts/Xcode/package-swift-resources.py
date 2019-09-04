#! /usr/bin/env python

# package-swift-resources.py
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

# package-swift-resources.py
#
# The Swift compiler depends on all the overlays and libraries for the Swift
# compiler.  These are version-matched with Swift.

# TARGET_DIR is where the lldb framework/shared library gets put.
# LLVM_BUILD_DIR is where LLVM and Clang got built
# LLVM_BUILD_DIR/lib/clang should exist and contain headers

import os
import re
import shutil
import sys

import lldbbuild

if len(sys.argv) != 3:
    print "usage: " + sys.argv[0] + " TARGET_DIR LLVM_BUILD_DIR"
    sys.exit(1)

if os.environ['LLVM_CONFIGURATION'] == 'BuildAndIntegration':
    print "Not copying Swift resources in B&I"
    sys.exit(0)

if os.environ['ACTION'] == 'install':
    print "Not copying Swift resources during installation"
    sys.exit(0)

target_dir = sys.argv[1]
swift_build_dir = lldbbuild.expected_package_build_path_for("swift")
print("JSMTMP lldbbuild build path for swift: %s" % swift_build_dir)

if not os.path.isdir(target_dir):
    print target_dir + " doesn't exist"
    sys.exit(1)

if not os.path.isdir(swift_build_dir):
    swift_build_dir = re.sub("-macosx-", "-iphoneos-", swift_build_dir)

if not os.path.isdir(swift_build_dir):
    swift_build_dir = re.sub("-iphoneos-", "-appletvos-", swift_build_dir)

if not os.path.isdir(swift_build_dir):
    swift_build_dir = re.sub("-appletvos-", "-watchos-", swift_build_dir)

if not os.path.isdir(swift_build_dir):
    print swift_build_dir + " doesn't exist"
    sys.exit(1)

resources = os.path.join(target_dir, "LLDB.framework", "Resources")

if not os.path.isdir(resources):
    print resources + " must exist"
    sys.exit(1)

swift_dir = os.path.join(swift_build_dir, "lib", "swift")

if not os.path.isdir(swift_dir):
    print swift_dir + " must exist"
    sys.exit(1)

# Just checking... we're actually going to copy all of swift_dir
shims_dir = os.path.join(swift_dir, "shims")

if not os.path.isdir(shims_dir):
    print shims_dir + " is not a directory"
    sys.exit(1)

swift_resources = os.path.join(resources, "Swift")

print("JSMTMP swift resources diretctory: %s" % swift_resources)

if os.path.isdir(swift_resources):
    shutil.rmtree(swift_resources)

shutil.copytree(
    swift_dir,
    swift_resources,
    ignore=shutil.ignore_patterns("install-tmp"))
