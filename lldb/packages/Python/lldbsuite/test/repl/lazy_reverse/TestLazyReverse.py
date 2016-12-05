# TestLazyReverse.py
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
"""Test that the REPL supports lazy reversing."""

import lldbsuite.test.lldbrepl as lldbrepl


class REPLLazyReverseTestCase(lldbrepl.REPLTest):

    mydir = lldbrepl.REPLTest.compute_mydir(__file__)

    def doTest(self):
        self.command(
            'var a = ["a","b","c"]; Array(a.lazy.reversed())',
            patterns=[
                r'\$R0: \[String] = 3 values {',
                r'\[0] = "c"',
                r'\[1] = "b"',
                r'\[2] = "a"'])
