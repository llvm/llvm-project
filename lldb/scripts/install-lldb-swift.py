#!/usr/bin/env python

# install-lldb-swift.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See http://swift.org/LICENSE.txt for license information
# See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------

# This script will install the files from an LLDB build into Xcode.

import argparse
import os
import shutil
import sys

parser = argparse.ArgumentParser(
    description="Installs the files from an LLDB build into Xcode.  Without options, tries to do the right thing.")
parser.add_argument("-l", "--lldb", help="Path to build/Configuration.")
parser.add_argument(
    "-s",
    "--swift",
    help="Path to llvm-build/LLVMConfig/arch/LLVMConfig.")
parser.add_argument("-x", "--xcode", help="Path to Xcode.app.")
parser.add_argument(
    "-d",
    "--dry",
    help="Dry run: log what would happen but don't do it.",
    action="store_true")

args = parser.parse_args()


def checkCandidates(candidates, checker):
    for candidate in candidates:
        if checker(candidate):
            return candidate
    return None

# Checker helpers


def checkDirOrLink(path):
    return (os.path.isdir(path) or os.path.islink(path))


def checkFileOrLink(path):
    return (os.path.isfile(path) or os.path.islink(path))

# Find an LLDB build


def checkLLDB(path):
    if not checkFileOrLink(path + "/lldb"):
        return False
    if not checkDirOrLink(path + "/LLDB.framework"):
        return False
    return True


def findLLDB():
    lldb_candidates = [
        "build/DebugClang",
        "build/Debug",
        "build/Release"
    ]
    return checkCandidates(lldb_candidates, checkLLDB)

# Find a Swift build


def checkSwift(path):
    if not checkFileOrLink(path + "/bin/swift"):
        return False
    if not checkDirOrLink(path + "/lib/swift"):
        return False
    return True


def findSwift():
    swift_candidates = [
        "llvm-build/Debug+Asserts/x86_64",
        "llvm-build/Debug/x86_64",
        "llvm-build/Release+Debug/x86_64",
        "llvm-build/Release+Asserts/x86_64",
        "llvm-build/Release/x86_64"
    ]
    return checkCandidates(swift_candidates, checkSwift)

# Find an Xcode installation


def checkXcode(path):
    if not checkFileOrLink(
            path +
            "/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift"):
        return False
    if not checkDirOrLink(
            path +
            "/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift"):
        return False
    return True


def findXcode():
    xcode_candidates = [
        "/Applications/Xcode.app"
    ]
    return checkCandidates(xcode_candidates, checkXcode)

# Check input arguments


def getPaths():
    lldb = None

    if args.lldb:
        lldb = args.lldb
    else:
        lldb = findLLDB()

    swift = None

    if args.swift:
        swift = args.swift
    else:
        swift = findSwift()

    xcode = None

    if args.xcode:
        xcode = args.xcode
    else:
        xcode = findXcode()

    if not lldb:
        print "Couldn't find LLDB automatically.  Please use --lldb."
        return None
    if not checkLLDB(lldb):
        print "The path passed to --lldb is not a valid LLDB build."
        return None

    if not swift:
        print "Couldn't find Swift automatically.  Please use --swift."
        return None
    if not checkSwift(swift):
        print "The path passed to --swift is not a valid Swift build."
        return None

    if not xcode:
        print "Couldn't find Xcode automatically.  Please use --xcode."
        return None
    if not checkXcode(xcode):
        print "The path passed to --xcode is not a valid Xcode installation."
        return None

    return (lldb, swift, xcode)

# Work unit classes


class WorkUnit:

    def __init__(self, from_path, to_path):
        self.from_path = from_path
        self.to_path = to_path
        self.backup_path = to_path + ".backup"

    # basic operations

    def remove(self, path):
        if os.path.islink(path):
            print "Remove alias " + self.to_path
        elif os.path.isdir(path):
            print "Remove (recursive) " + self.to_path
        else:
            print "Remove " + self.to_path
        if not args.dry:
            if os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    def removeBackup(self):
        self.remove(self.backup_path)

    def backupUnchecked(self):
        print "Move " + self.to_path + " to " + self.backup_path
        if not args.dry:
            os.rename(self.to_path, self.backup_path)

    def removeTo(self):
        self.remove(self.to_path)

    def linkFromTo(self):
        abs_from_path = os.path.abspath(self.from_path)
        print "Link " + abs_from_path + " to " + self.to_path
        if not args.dry:
            os.symlink(abs_from_path, self.to_path)

    # high-level operations

    def checkAndBackup(self):
        if os.path.islink(self.to_path):
            self.removeTo()  # no backup necessary
        else:
            if os.path.exists(self.backup_path):
                self.removeBackup()
            self.backupUnchecked()

    def install(self):
        self.linkFromTo()

# Make path dictionary


def makeWorkUnits(lldb, swift, xcode):
    toolchain = xcode + "/Contents/Developer/Toolchains/XcodeDefault.xctoolchain"
    toolchain_usr_bin = toolchain + "/usr/bin"
    toolchain_usr_lib = toolchain + "/usr/lib"

    work_units = [
        WorkUnit(
            swift +
            "/bin/swift",
            toolchain_usr_bin +
            "/swift"),
        WorkUnit(
            swift +
            "/lib/swift",
            toolchain_usr_lib +
            "/swift"),
        WorkUnit(
            lldb +
            "/LLDB.framework",
            xcode +
            "/Contents/SharedFrameworks/LLDB.framework")]

    # if we've built sourcekitd, install that too

#  ## commented out because a built sourcekit doesn't work
#
#  if os.path.isdir(swift + "/lib/sourcekitd.framework"):
#    work_units.append(WorkUnit(swift + "/lib/sourcekitd.framework", toolchain_usr_lib + "/sourcekitd.framework"))

    return work_units

# Prepare Xcode installation, backing up data as necessary


def prepareAndBackup(work_units):
    for work_unit in work_units:
        work_unit.checkAndBackup()

# Install


def install(work_units):
    for work_unit in work_units:
        work_unit.install()

# Main

validated_paths = getPaths()

if not validated_paths:
    sys.exit(0)

(lldb, swift, xcode) = validated_paths

print "Installing LLDB from " + lldb + " and Swift from " + swift + " into Xcode at " + xcode

work_units = makeWorkUnits(lldb, swift, xcode)

prepareAndBackup(work_units)
install(work_units)
