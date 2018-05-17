# TestPlaygrounds.py
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
"""
Test that playgrounds work
"""
import commands
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import platform
import unittest2


def execute_command(command):
    (exit_status, output) = commands.getstatusoutput(command)
    return exit_status


class TestSwiftPlaygrounds(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.skipIf(
        debug_info=decorators.no_match("dsym"),
        bugnumber="This test only builds one way",
        macos_version=["<", "10.11"])
    @decorators.add_test_categories(["swiftpr"])
    def test_cross_module_extension(self):
        """Test that playgrounds work"""
        self.build()
        self.do_test(True)
        self.do_test(False)

    def setUp(self):
        TestBase.setUp(self)
        self.PlaygroundStub_source = "PlaygroundStub.swift"
        self.PlaygroundStub_source_spec = lldb.SBFileSpec(
            self.PlaygroundStub_source)

    def do_test(self, force_target):
        """Test that playgrounds work"""
        exe_name = "PlaygroundStub"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        if force_target:
            version, _, machine = platform.mac_ver()
            triple = '%s-apple-macosx%s' % (machine, version)
            target = self.dbg.CreateTargetWithFileAndArch(exe, triple)
        else:
            target = self.dbg.CreateTarget(exe)
            
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', self.PlaygroundStub_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        contents = ""

        with open('Contents.swift', 'r') as contents_file:
            contents = contents_file.read()

        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeSwift)
        options.SetPlaygroundTransformEnabled()

        self.frame.EvaluateExpression(contents, options)

        ret = self.frame.EvaluateExpression("get_output()")

        playground_output = ret.GetSummary()
        if not force_target:
            # This is expected to fail because the deployment target
            # is less than the availability of the function being
            # called.
            self.assertTrue(playground_output == '""')
            return

        self.assertTrue(playground_output is not None)
        self.assertTrue("a=\\'3\\'" in playground_output)
        self.assertTrue("b=\\'5\\'" in playground_output)
        self.assertTrue("=\\'8\\'" in playground_output)
        self.assertTrue("=\\'11\\'" in playground_output)

       
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
