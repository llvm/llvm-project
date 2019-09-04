# TestLastLineAssignmentLog.py
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
Test that an assignment made as the last line in a Playground is properly logged
"""
import lldbsuite.test.lldbplaygroundrepl as repl
from lldbsuite.test.lldbtest import *


class TestLastLineAssignmentLog(repl.PlaygroundREPLTest):

    mydir = repl.PlaygroundREPLTest.compute_mydir(__file__)

    def do_test(self):
        """
        Test that statements made in one block can be referenced in a
        proceeding block
        """

        # Execute first block
        result, output = self.execute_code("Input1.swift")
        playground_output = output.GetSummary()

        if self.is_compile_or_runtime_error(result):
            self.did_crash(result)
            self.assertTrue(False)

        self.assertTrue(playground_output is not None)
        self.assertTrue("a=\\'1\\'" in playground_output)
        self.assertTrue("=\\'2\\'" in playground_output)

        # Execute second block
        result, output = self.execute_code("Input2.swift")
        playground_output = output.GetSummary()

        if self.is_compile_or_runtime_error(result):
            self.did_crash(result)
            self.assertTrue(False)

        self.assertTrue(playground_output is not None)
        self.assertTrue("=\\'3\\'" in playground_output)
