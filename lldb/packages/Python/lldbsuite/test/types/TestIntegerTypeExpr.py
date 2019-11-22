"""
Test that variable expressions of type integer are evaluated correctly.
"""

import AbstractBase

from lldbsuite.test.decorators import *


class IntegerTypeExprTestCase(AbstractBase.GenericTester):

    mydir = AbstractBase.GenericTester.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_unsigned_short_type_from_block(self):
        """Test that 'unsigned short'-type variables are displayed correctly from a block."""
        self.build_and_run_expr(
            'unsigned_short.cpp', set(['unsigned', 'short']), bc=True)

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
