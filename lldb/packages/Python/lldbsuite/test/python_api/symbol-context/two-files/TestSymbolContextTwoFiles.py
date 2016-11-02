"""
Test SBSymbolContext APIs.
"""

from __future__ import print_function

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SymbolContextTwoFilesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"])
    def test_lookup_by_address(self):
        """Test lookup by address in a module with multiple compilation units"""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.IsValid())
        for symbol_name in ["struct1::f()", "struct2::f()"]:
            sc_list = module.FindFunctions(symbol_name, lldb.eSymbolTypeCode)
            self.assertTrue(1, sc_list.GetSize())
            symbol_address = sc_list.GetContextAtIndex(
                0).GetSymbol().GetStartAddress()
            self.assertTrue(symbol_address.IsValid())
            sc_by_address = module.ResolveSymbolContextForAddress(
                symbol_address, lldb.eSymbolContextFunction)
            self.assertEqual(symbol_name,
                             sc_by_address.GetFunction().GetName())
