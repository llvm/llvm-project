"""
Test symbol table access for main.m.
"""

from __future__ import print_function


import os
import time

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
class FoundationSymtabTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    symbols_list = ['-[MyString initWithNSString:]',
                    '-[MyString dealloc]',
                    '-[MyString description]',
                    '-[MyString descriptionPauses]',     # synthesized property
                    # synthesized property
                    '-[MyString setDescriptionPauses:]',
                    'Test_Selector',
                    'Test_NSString',
                    'Test_MyString',
                    'Test_NSArray',
                    'main'
                    ]

    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Test symbol table access with Python APIs."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        #
        # Exercise Python APIs to access the symbol table entries.
        #

        # Create the filespec by which to locate our a.out module.
        filespec = lldb.SBFileSpec(exe, False)

        module = target.FindModule(filespec)
        self.assertTrue(module, VALID_MODULE)

        # Create the set of known symbols.  As we iterate through the symbol
        # table, remove the symbol from the set if it is a known symbol.
        expected_symbols = set(self.symbols_list)
        for symbol in module:
            self.assertTrue(symbol, VALID_SYMBOL)
            #print("symbol:", symbol)
            name = symbol.GetName()
            if name in expected_symbols:
                #print("Removing %s from known_symbols %s" % (name, expected_symbols))
                expected_symbols.remove(name)

        # At this point, the known_symbols set should have become an empty set.
        # If not, raise an error.
        #print("symbols unaccounted for:", expected_symbols)
        self.assertTrue(len(expected_symbols) == 0,
                        "All the known symbols are accounted for")
