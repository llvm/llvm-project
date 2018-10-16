"""
Test that objective-c constant strings are generated correctly by the expression
parser.
"""

from __future__ import print_function


import os
import time
import shutil
import subprocess
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
class TestObjCBreakpoints(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_break(self):
        """Test setting Objective-C specific breakpoints (DWARF in .o files)."""
        self.build()
        self.setTearDownCleanup()
        self.check_objc_breakpoints(False)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.main_source = "main.m"
        self.line = line_number(self.main_source, '// Set breakpoint here')

    def check_category_breakpoints(self):
        name_bp = self.target.BreakpointCreateByName("myCategoryFunction")
        selector_bp = self.target.BreakpointCreateByName(
            "myCategoryFunction",
            lldb.eFunctionNameTypeSelector,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList())
        self.assertTrue(
            name_bp.GetNumLocations() == selector_bp.GetNumLocations(),
            'Make sure setting a breakpoint by name "myCategoryFunction" sets a breakpoint even though it is in a category')
        for bp_loc in selector_bp:
            function_name = bp_loc.GetAddress().GetSymbol().GetName()
            self.assertTrue(
                " myCategoryFunction]" in function_name,
                'Make sure all function names have " myCategoryFunction]" in their names')

        category_bp = self.target.BreakpointCreateByName(
            "-[MyClass(MyCategory) myCategoryFunction]")
        stripped_bp = self.target.BreakpointCreateByName(
            "-[MyClass myCategoryFunction]")
        stripped2_bp = self.target.BreakpointCreateByName(
            "[MyClass myCategoryFunction]")
        self.assertTrue(
            category_bp.GetNumLocations() == 1,
            "Make sure we can set a breakpoint using a full objective C function name with the category included (-[MyClass(MyCategory) myCategoryFunction])")
        self.assertTrue(
            stripped_bp.GetNumLocations() == 1,
            "Make sure we can set a breakpoint using a full objective C function name without the category included (-[MyClass myCategoryFunction])")
        self.assertTrue(
            stripped2_bp.GetNumLocations() == 1,
            "Make sure we can set a breakpoint using a full objective C function name without the category included ([MyClass myCategoryFunction])")

    def check_objc_breakpoints(self, have_dsym):
        """Test constant string generation amd comparison by the expression parser."""

        # Set debugger into synchronous mode
        self.dbg.SetAsync(False)

        # Create a target by the debugger.
        exe = self.getBuildArtifact("a.out")
        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        #----------------------------------------------------------------------
        # Set breakpoints on all selectors whose name is "count". This should
        # catch breakpoints that are both C functions _and_ anything whose
        # selector is "count" because just looking at "count" we can't tell
        # definitively if the name is a selector or a C function
        #----------------------------------------------------------------------
        name_bp = self.target.BreakpointCreateByName("count")
        selector_bp = self.target.BreakpointCreateByName(
            "count",
            lldb.eFunctionNameTypeSelector,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList())
        self.assertTrue(
            name_bp.GetNumLocations() >= selector_bp.GetNumLocations(),
            'Make sure we get at least the same amount of breakpoints if not more when setting by name "count"')
        self.assertTrue(
            selector_bp.GetNumLocations() > 50,
            'Make sure we find a lot of "count" selectors')  # There are 93 on the latest MacOSX
        for bp_loc in selector_bp:
            function_name = bp_loc.GetAddress().GetSymbol().GetName()
            self.assertTrue(
                " count]" in function_name,
                'Make sure all function names have " count]" in their names')

        #----------------------------------------------------------------------
        # Set breakpoints on all selectors whose name is "isEqual:". This should
        # catch breakpoints that are only ObjC selectors because no C function
        # can end with a :
        #----------------------------------------------------------------------
        name_bp = self.target.BreakpointCreateByName("isEqual:")
        selector_bp = self.target.BreakpointCreateByName(
            "isEqual:",
            lldb.eFunctionNameTypeSelector,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList())
        self.assertTrue(
            name_bp.GetNumLocations() == selector_bp.GetNumLocations(),
            'Make sure setting a breakpoint by name "isEqual:" only sets selector breakpoints')
        for bp_loc in selector_bp:
            function_name = bp_loc.GetAddress().GetSymbol().GetName()
            self.assertTrue(
                " isEqual:]" in function_name,
                'Make sure all function names have " isEqual:]" in their names')

        self.check_category_breakpoints()

        if have_dsym:
            shutil.rmtree(exe + ".dSYM")
        self.assertTrue(subprocess.call(
            ['/usr/bin/strip', '-Sx', exe]) == 0, 'stripping dylib succeeded')

        # Check breakpoints again, this time using the symbol table only
        self.check_category_breakpoints()
