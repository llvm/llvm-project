#! /usr/bin/env python

# package-clang-headers.py
#
# The Clang module loader depends on built-in headers for the Clang compiler.
# We grab these from the Clang build and move them into the LLDB module.

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

target_dir = sys.argv[1]
llvm_build_dir = lldbbuild.expected_package_build_path_for("llvm")

if not os.path.isdir(target_dir):
    print target_dir + " doesn't exist"
    sys.exit(1) 

if not os.path.isdir(llvm_build_dir):
    llvm_build_dir = re.sub ("-macosx-", "-iphoneos-", llvm_build_dir)

if not os.path.isdir(llvm_build_dir):
    llvm_build_dir = re.sub ("-iphoneos-", "-appletvos-", llvm_build_dir)

if not os.path.isdir(llvm_build_dir):
    llvm_build_dir = re.sub ("-appletvos-", "-watchos-", llvm_build_dir)

if not os.path.isdir(llvm_build_dir):
    print llvm_build_dir + " doesn't exist"
    sys.exit(1)

resources = os.path.join(target_dir, "LLDB.framework", "Resources")

if not os.path.isdir(resources):
    print resources + " must exist"
    sys.exit(1)

clang_dir = os.path.join(llvm_build_dir, "lib", "clang")

if not os.path.isdir(clang_dir):
    print clang_dir + " must exist"
    sys.exit(1)

version_dir = None

for subdir in os.listdir(clang_dir):
    if (re.match("^[0-9]+(\.[0-9]+)*$", subdir)):
        version_dir = os.path.join(clang_dir, subdir)
        break

if version_dir == None:
    print "Couldn't find a subdirectory of the form #(.#)... in " + clang_dir
    sys.exit(1)

if not os.path.isdir(version_dir):
    print version_dir + " is not a directory"
    sys.exit(1)

# Just checking... we're actually going to copy all of version_dir
include_dir = os.path.join(version_dir, "include")

if not os.path.isdir(include_dir):
    print version_dir + " is not a directory"
    sys.exit(1)

clang_resources = os.path.join(resources, "Clang")

if os.path.isdir(clang_resources):
    shutil.rmtree(clang_resources)

shutil.copytree(version_dir, clang_resources)
