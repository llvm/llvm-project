"""
Test that variables of type short are displayed correctly.
"""

import AbstractBase

from lldbsuite.test.decorators import *


class ShortTypeTestCase(AbstractBase.GenericTester):

    mydir = AbstractBase.GenericTester.compute_mydir(__file__)

    def test_short_type(self):
        """Test that short-type variables are displayed correctly."""
        self.build_and_run('short.cpp', set(['short']))

    @skipUnlessDarwin
    def test_short_type_from_block(self):
        """Test that short-type variables are displayed correctly from a block."""
        self.build_and_run('short.cpp', set(['short']), bc=True)

    def test_unsigned_short_type(self):
        """Test that 'unsigned_short'-type variables are displayed correctly."""
        self.build_and_run('unsigned_short.cpp', set(['unsigned', 'short']))

    @skipUnlessDarwin
    def test_unsigned_short_type_from_block(self):
        """Test that 'unsigned short'-type variables are displayed correctly from a block."""
        self.build_and_run(
            'unsigned_short.cpp', set(['unsigned', 'short']), bc=True)
