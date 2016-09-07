"""
Test that variable expressions of integer basic types are evaluated correctly.
"""

from __future__ import print_function


import AbstractBase
import sys

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class IntegerTypesExprTestCase(AbstractBase.GenericTester):

    mydir = AbstractBase.GenericTester.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        AbstractBase.GenericTester.setUp(self)
        # disable "There is a running process, kill it and restart?" prompt
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(
            lambda: self.runCmd("settings clear auto-confirm"))

    def test_char_type(self):
        """Test that char-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('char.cpp', set(['char']), qd=True)

    @skipUnlessDarwin
    def test_char_type_from_block(self):
        """Test that char-type variables are displayed correctly from a block."""
        self.build_and_run_expr('char.cpp', set(['char']), bc=True, qd=True)

    def test_unsigned_char_type(self):
        """Test that 'unsigned_char'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_char.cpp', set(
            ['unsigned', 'char']), qd=True)

    @skipUnlessDarwin
    def test_unsigned_char_type_from_block(self):
        """Test that 'unsigned char'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('unsigned_char.cpp', set(
            ['unsigned', 'char']), bc=True, qd=True)

    def test_short_type(self):
        """Test that short-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('short.cpp', set(['short']))

    @skipUnlessDarwin
    def test_short_type_from_block(self):
        """Test that short-type variables are displayed correctly from a block."""
        self.build_and_run_expr('short.cpp', set(['short']), bc=True)

    def test_unsigned_short_type(self):
        """Test that 'unsigned_short'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_short.cpp',
                                set(['unsigned', 'short']))

    @skipUnlessDarwin
    def test_unsigned_short_type_from_block(self):
        """Test that 'unsigned short'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('unsigned_short.cpp', set(
            ['unsigned', 'short']), bc=True)

    def test_int_type(self):
        """Test that int-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('int.cpp', set(['int']))

    @skipUnlessDarwin
    def test_int_type_from_block(self):
        """Test that int-type variables are displayed correctly from a block."""
        self.build_and_run_expr('int.cpp', set(['int']))

    def test_unsigned_int_type(self):
        """Test that 'unsigned_int'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_int.cpp', set(['unsigned', 'int']))

    @skipUnlessDarwin
    def test_unsigned_int_type_from_block(self):
        """Test that 'unsigned int'-type variables are displayed correctly from a block."""
        self.build_and_run_expr(
            'unsigned_int.cpp', set(['unsigned', 'int']), bc=True)

    def test_long_type(self):
        """Test that long-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('long.cpp', set(['long']))

    @skipUnlessDarwin
    def test_long_type_from_block(self):
        """Test that long-type variables are displayed correctly from a block."""
        self.build_and_run_expr('long.cpp', set(['long']), bc=True)

    def test_unsigned_long_type(self):
        """Test that 'unsigned long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_long.cpp', set(['unsigned', 'long']))

    @skipUnlessDarwin
    def test_unsigned_long_type_from_block(self):
        """Test that 'unsigned_long'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('unsigned_long.cpp', set(
            ['unsigned', 'long']), bc=True)

    def test_long_long_type(self):
        """Test that 'long long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('long_long.cpp', set(['long long']))

    @skipUnlessDarwin
    def test_long_long_type_from_block(self):
        """Test that 'long_long'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('long_long.cpp', set(['long long']), bc=True)

    def test_unsigned_long_long_type(self):
        """Test that 'unsigned long long'-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('unsigned_long_long.cpp',
                                set(['unsigned', 'long long']))

    @skipUnlessDarwin
    def test_unsigned_long_long_type_from_block(self):
        """Test that 'unsigned_long_long'-type variables are displayed correctly from a block."""
        self.build_and_run_expr('unsigned_long_long.cpp', set(
            ['unsigned', 'long long']), bc=True)
