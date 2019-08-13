# TestSwiftDWARFImporterC.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2019 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftDWARFImporterC(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def build(self):
        include = self.getBuildArtifact('include')
        inputs = self.getSourcePath('Inputs')
        lldbutil.mkdir_p(include)
        import shutil
        for f in ['module.modulemap', 'c-header.h']:
            shutil.copyfile(os.path.join(inputs, f), os.path.join(include, f))

        super(TestSwiftDWARFImporterC, self).build()

        # Remove the header files to thwart ClangImporter.
        self.assertTrue(os.path.isdir(include))
        shutil.rmtree(include)

    @skipIf(archs=['ppc64le'])
    # SR-10214
    @swiftTest
    # This test needs a working Remote Mirrors implementation.
    @skipIf(oslist=['linux', 'windows'])
    def test_dwarf_importer(self):
        lldb.SBDebugger.MemoryPressureDetected()
        self.runCmd("settings set symbols.use-swift-dwarfimporter true")
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        lldbutil.check_variable(self,
                                target.FindFirstGlobalVariable("pureSwift"),
                                value="42")
        lldbutil.check_variable(self,
                                target.FindFirstGlobalVariable("point"),
                                typename='__ObjC.Point', num_children=2)
        self.expect("fr v point", substrs=["x = 1", "y = 2"])
        self.expect("fr v point", substrs=["x = 1", "y = 2"])
        self.expect("fr v enumerator", substrs=[".yellow"])
        self.expect("fr v pureSwiftStruct", substrs=["pure swift"])
        self.expect("fr v swiftStructCMember",
                    substrs=["x = 3", "y = 4", "swift struct c member"])
        self.expect("fr v typedef", substrs=["x = 5", "y = 6"])
        self.expect("fr v union", substrs=["(DoubleLongUnion)", "long_val = 42"])
        process.Clear()
        target.Clear()
        lldb.SBDebugger.MemoryPressureDetected()

    @skipIf(archs=['ppc64le'])
    # SR-10214
    @swiftTest
    def test_negative(self):
        lldb.SBDebugger.MemoryPressureDetected()
        self.runCmd("log enable lldb types")
        self.runCmd("settings set symbols.use-swift-dwarfimporter false")
        self.build()
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        lldbutil.check_variable(self,
                                target.FindFirstGlobalVariable("point"),
                                typename="Point", num_children=2)
        self.expect("fr v point", substrs=["x = 1", "y = 2"])

        found = False
        logfile = open(log, "r")
        for line in logfile:
            if "missing required module":
                found = True
        self.assertTrue(found)

        process.Clear()
        target.Clear()
        lldb.SBDebugger.MemoryPressureDetected()
        self.runCmd("settings set symbols.use-swift-dwarfimporter true")
       
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
