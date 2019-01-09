# TestSwiftDWARFImporter.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2018 Apple Inc. and the Swift project authors
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


class TestSwiftDWARFImporter(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @swiftTest
    def test_dwarf_importer(self):
        self.runCmd("settings set symbols.use-swift-dwarfimporter true")
        
        include = self.getBuildArtifact('include')
        inputs = self.getSourcePath('Inputs')
        lldbutil.mkdir_p(include)
        import shutil
        for f in ['module.modulemap', 'objc-header.h']:
            shutil.copyfile(os.path.join(inputs, f), os.path.join(include, f))

        self.build()

        # Remove the header files to thwart ClangImporter.
        self.assertTrue(os.path.isdir(include))
        shutil.rmtree(include)
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        lldbutil.check_variable(self,
                                target.FindFirstGlobalVariable("pureSwift"),
                                value="42")
        lldbutil.check_variable(self,
                                target.FindFirstGlobalVariable("point"),
                                typename="Point", num_children=2)
        self.expect("fr v point", substrs=["x = 1", "y = 2"])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
