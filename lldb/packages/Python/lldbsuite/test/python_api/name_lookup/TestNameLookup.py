"""
Test SBTarget APIs.
"""

from __future__ import print_function


import unittest2
import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestNameLookup(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"], bugnumber='llvm.org/pr21765')
    def test_target(self):
        """Exercise SBTarget.FindFunctions() with various name masks.
        
        A previous regression caused mangled names to not be able to be looked up.
        This test verifies that using a mangled name with eFunctionNameTypeFull works
        and that using a function basename with eFunctionNameTypeFull works for all
        C++ functions that are at the global namespace level."""
        self.build();
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        exe_module = target.FindModule(target.GetExecutable())
        
        c_name_to_symbol = {}
        cpp_name_to_symbol = {}
        mangled_to_symbol = {}
        num_symbols = exe_module.GetNumSymbols();
        for i in range(num_symbols):
            symbol = exe_module.GetSymbolAtIndex(i);
            name = symbol.GetName()
            if name and 'unique_function_name' in name and '__PRETTY_FUNCTION__' not in name:
                mangled = symbol.GetMangledName()
                if mangled:
                    mangled_to_symbol[mangled] = symbol
                    if name:
                        cpp_name_to_symbol[name] = symbol
                elif name:
                    c_name_to_symbol[name] = symbol

        # Make sure each mangled name turns up exactly one match when looking up
        # functions by full name and using the mangled name as the name in the 
        # lookup
        self.assertGreaterEqual(len(mangled_to_symbol), 6)
        for mangled in mangled_to_symbol.keys():
            symbol_contexts = target.FindFunctions(mangled, lldb.eFunctionNameTypeFull)
            self.assertTrue(symbol_contexts.GetSize() == 1)
            for symbol_context in symbol_contexts:
                self.assertTrue(symbol_context.GetFunction().IsValid())
                self.assertTrue(symbol_context.GetSymbol().IsValid())
            
            
