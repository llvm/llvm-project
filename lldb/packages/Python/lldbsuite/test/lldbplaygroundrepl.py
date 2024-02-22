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
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import unittest2

import sys
if sys.version_info.major == 2:
    import commands as subprocess
else:
    import subprocess

class PlaygroundREPLTest(TestBase):

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.skipIf(
        debug_info=decorators.no_match("dsym"),
        bugnumber="This test only builds one way")

    def build_all(self):
        self.build()

    def execute_command(self, command):
        (exit_status, output) = subprocess.getstatusoutput(command)
        return exit_status

    def repl_set_up(self):
        """
        Playgrounds REPL test specific setup that must happen after class setup
        """
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('PlaygroundStub.swift'),
            exe_name='PlaygroundStub', extra_images=['libPlaygroundsRuntime.dylib'])

        self.frame = thread.frames[0]
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
        output = self.frame.EvaluateExpression("get_output()")
        with recording(self, self.TraceOn()) as sbuf:
            print("playground result: ", file=sbuf)
            print(str(result), file=sbuf)
            print("playground output:", file=sbuf)
            print(str(output), file=sbuf)
        self.assertSuccess(output.GetError())

        return result, output

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

    @swiftTest
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
