# lldbplaygroundrepl.py
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
import lldb
import commands
from lldbtest import *
import decorators
import lldbutil
import os
import os.path
import unittest2


class PlaygroundREPLTest(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.skipIf(
        debug_info=decorators.no_match("dsym"),
        bugnumber="This test only builds one way")

    def build_all(self):
        self.build()

    def execute_command(self, command):
        (exit_status, output) = commands.getstatusoutput(command)
        return exit_status

    def setUp(self):
        TestBase.setUp(self)
        self.PlaygroundStub_source = "PlaygroundStub.swift"
        self.PlaygroundStub_source_spec = lldb.SBFileSpec(
            self.PlaygroundStub_source)

    def repl_set_up(self):
        """
        Playgrounds REPL test specific setup that must happen after class setup
        """
        exe_name = "PlaygroundStub"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', self.PlaygroundStub_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        process = target.LaunchSimple(None, None, self.getBuildDir())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        # Configure lldb
        self.options = lldb.SBExpressionOptions()
        self.options.SetLanguage(lldb.eLanguageTypeSwift)
        self.options.SetPlaygroundTransformEnabled()
        self.options.SetREPLMode()

        # Add cleanup to tear down
        def cleanup():
            self.execute_command("make cleanup")
        self.addTearDownHook(cleanup)

    def execute_code(self, inputFile):
        """
        Execute a block of code contained within a Swift file. Requires a
        frame to have already been created.

        :param inputFile: file name (with extension) of code contents
        :return: result: resulting SBStream from executing playground
                 output: string summary of playground execution
        """

        # Execute code
        contents = ""

        with open(inputFile, 'r') as contents_file:
            contents = contents_file.read()

        result = self.frame.EvaluateExpression(contents, self.options)
        ouput = self.frame.EvaluateExpression("get_output()")

        return result, ouput

    def is_compile_or_runtime_error(self, result):
        """
        Determine if any errors we care about for Playground execution occurred

        :param result: SBStream from executing a playground
        :return: ret: bool value of if it's an error we care about
        """
        ret = result.GetError().Fail() and not (
                result.GetError().GetType() == 1 and
                result.GetError().GetError() == 0x1001)
        return ret

    def get_stream_data(self, result):
        stream = lldb.SBStream()
        stream.Clear()
        result.GetError().GetDescription(stream)
        data = stream.GetData()
        return data

    def did_crash(self, result):
        error = self.get_stream_data(result)
        print("Crash Error: {}".format(error))

    def test_playgrounds(self):
        # Build
        self.build_all()
        # Prepare
        self.repl_set_up()
        # Run user test
        self.do_test()

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
