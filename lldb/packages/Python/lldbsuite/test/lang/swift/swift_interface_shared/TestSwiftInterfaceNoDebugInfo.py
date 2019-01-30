# TestSwiftInterfaceNoDebugInfo.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2019 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# -----------------------------------------------------------------------------
"""
Test that we load and handle modules that only have textual .swiftinterface
files -- i.e. no associated .swiftmodule file -- and no debug info. The module
loader should generate the .swiftmodule for any .swiftinterface it finds unless
it is already in the module cache.
"""

import commands
import glob
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import unittest2


class TestSwiftInterfaceNoDebugInfo(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_swift_interface(self):
        """Test that we load and handle modules that only have textual .swiftinterface files"""
        self.build()
        self.do_test()


    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)


    def do_test(self):
        # A custom module cache location
        mod_cache = self.getBuildArtifact("module-cache-dir")

        # Clear the module cache if it already exists
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)
        self.assertFalse(os.path.isdir(mod_cache),
                         "module cache should not exist")

        # Update the settings to use the custom module cache location
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"' % mod_cache)

        # Set a breakpoint in and launch the main executable
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                "break here", self.main_source_spec,
                exe_name="main")

        self.frame = thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        # Check we are able to access the public fields of variables whose
        # types are from the .swiftinterface-only dylibs
        var = self.frame.FindVariable("x")
        lldbutil.check_variable(self, var, False, typename="AA.MyPoint")

        child_y = var.GetChildMemberWithName("y") # MyPoint.y is public
        lldbutil.check_variable(self, child_y, False, value="0")

        child_x = var.GetChildMemberWithName("x") # MyPoint.x isn't public
        self.assertFalse(child_x.IsValid())

        # Expression evaluation using types from the .swiftinterface only
        # dylibs should work too
        lldbutil.check_expression(self, self.frame, "y.magnitudeSquared", "404", use_summary=False)
        lldbutil.check_expression(self, self.frame, "MyPoint(x: 1, y: 2).magnitudeSquared", "5", use_summary=False)

        # Check the module cache was populated with the .swiftmodule files of
        # the loaded modules
        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")
        a_modules = glob.glob(os.path.join(mod_cache, 'AA-*.swiftmodule'))
        b_modules = glob.glob(os.path.join(mod_cache, 'BB-*.swiftmodule'))
        c_modules = glob.glob(os.path.join(mod_cache, 'CC-*.swiftmodule'))
        self.assertEqual(len(a_modules), 1)
        self.assertEqual(len(b_modules), 1)
        self.assertEqual(len(c_modules), 0)

        # Update the timestamps of the modules to a time well in the past
        for file in a_modules + b_modules:
            make_old(file)

        # Re-import module A and B
        self.runCmd("expr import AA")
        self.runCmd("expr import BB")

        # Import C for the first time and check we can evaluate expressions
        # involving types from it
        self.runCmd("expr import CC")
        lldbutil.check_expression(self, self.frame, "Baz.baz()", "23", use_summary=False)

        # Check we still have a single .swiftmodule in the cache for A and B
        # and that there is now one for C too
        a_modules = glob.glob(os.path.join(mod_cache, 'AA-*.swiftmodule'))
        b_modules = glob.glob(os.path.join(mod_cache, 'BB-*.swiftmodule'))
        c_modules = glob.glob(os.path.join(mod_cache, 'CC-*.swiftmodule'))
        self.assertEqual(len(a_modules), 1, "unexpected number of swiftmodules for A.swift")
        self.assertEqual(len(b_modules), 1, "unexpected number of swiftmodules for B.swift")
        self.assertEqual(len(c_modules), 1, "unexpected number of swiftmodules for C.swift")

        # Make sure the .swiftmodule files of A and B were re-used rather than
        # re-generated when they were re-imported
        for file in a_modules + b_modules:
            self.assertTrue(is_old(file), "Swiftmodule file was regenerated rather than reused")


OLD_TIMESTAMP = 1390550700 # 2014-01-24T08:05:00+00:00

def make_old(file):
    """Sets the access and modified time of the given file to a time long past"""
    os.utime(file, (OLD_TIMESTAMP, OLD_TIMESTAMP))

def is_old(file):
    """Checks the modified time of the given file matches the timestamp set my make_old"""
    return os.stat(file).st_mtime == OLD_TIMESTAMP


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
