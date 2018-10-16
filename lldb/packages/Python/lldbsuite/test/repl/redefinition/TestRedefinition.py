# TestREPLBreakpoints.py
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
"""Test that redefining things in the repl works as desired."""

import lldbsuite.test.lldbrepl as lldbrepl

class TestRedefinition (lldbrepl.REPLTest):

    mydir = lldbrepl.REPLTest.compute_mydir(__file__)

    def doTest(self):
        # Define some *DefinedInREPL things.
        self.command('var varDefinedInREPL: Int = 1')
        self.command('func funcDefinedInREPL() -> Int { return 1 }')
        self.command('struct StructDefinedInREPL { var field: Int = 1 }')

        # Assert that the REPL can refer to the *DefinedInRepl things.
        self.command('varDefinedInREPL', patterns=[r'\$R[0-9]+: Int = 1'])
        self.command('funcDefinedInREPL()', patterns=[r'\$R[0-9]+: Int = 1'])
        self.command('StructDefinedInREPL().field',
                     patterns=[r'\$R[0-9]+: Int = 1'])

        # Redefine the *DefinedInREPL things.
        self.command('var varDefinedInREPL: Int = 2')
        self.command('func funcDefinedInREPL() -> Int { return 2 }')
        self.command('struct StructDefinedInREPL { var field: Int = 2 }')

        # Assert that the REPL refers to the redefinitions of the
        # *DefinedInREPL things.
        self.command('varDefinedInREPL', patterns=[r'\$R[0-9]+: Int = 2'])
        self.command('funcDefinedInREPL()', patterns=[r'\$R[0-9]+: Int = 2'])
        self.command('StructDefinedInREPL().field',
                     patterns=[r'\$R[0-9]+: Int = 2'])

        # Redefine the *DefinedInREPL things, and refer to them in the same
        # unit of code. Assert that the references refer to the new
        # redefinitions.
        self.command('''
                     var varDefinedInREPL: Int = 3
                     varDefinedInREPL
                     ''', patterns=[r'\$R[0-9]+: Int = 3'])
        self.command('''
                     func funcDefinedInREPL() -> Int { return 3 }
                     funcDefinedInREPL()
                     ''', patterns=[r'\$R[0-9]+: Int = 3'])
        self.command('''
                     struct StructDefinedInREPL { var field: Int = 3 }
                     StructDefinedInREPL().field
                     ''', patterns=[r'\$R[0-9]+: Int = 3'])
