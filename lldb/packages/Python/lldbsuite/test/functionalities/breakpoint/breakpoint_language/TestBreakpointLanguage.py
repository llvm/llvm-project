"""
Test that the language option for breakpoints works correctly
parser.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
import shutil
import subprocess


class TestBreakpointLanguage(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().

    def check_location_file(self, bp, loc, test_name):
        bp_loc = bp.GetLocationAtIndex(loc)
        addr = bp_loc.GetAddress()
        comp_unit = addr.GetCompileUnit()
        comp_name = comp_unit.GetFileSpec().GetFilename()
        return comp_name == test_name

    def test_regex_breakpoint_language(self):
        """Test that the name regex breakpoint commands obey the language filter."""

        self.build()
        # Create a target by the debugger.
        exe = os.path.join(os.getcwd(), "a.out")
        error = lldb.SBError()
        # Don't read in dependencies so we don't come across false matches that
        # add unwanted breakpoint hits.
        self.target = self.dbg.CreateTarget(exe, None, None, False, error)
        self.assertTrue(self.target, VALID_TARGET)

        cpp_bp = self.target.BreakpointCreateByRegex(
            "func_from",
            lldb.eLanguageTypeC_plus_plus,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList())
        self.assertTrue(
            cpp_bp.GetNumLocations() == 1,
            "Only one C++ symbol matches")
        self.assertTrue(self.check_location_file(cpp_bp, 0, "b.cpp"))

        c_bp = self.target.BreakpointCreateByRegex(
            "func_from",
            lldb.eLanguageTypeC,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList())
        self.assertTrue(
            c_bp.GetNumLocations() == 1,
            "Only one C symbol matches")
        self.assertTrue(self.check_location_file(c_bp, 0, "a.c"))

        objc_bp = self.target.BreakpointCreateByRegex(
            "func_from",
            lldb.eLanguageTypeObjC,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList())
        self.assertTrue(
            objc_bp.GetNumLocations() == 0,
            "No ObjC symbol matches")

    def test_by_name_breakpoint_language(self):
        """Test that the name regex breakpoint commands obey the language filter."""

        self.build()
        # Create a target by the debugger.
        exe = os.path.join(os.getcwd(), "a.out")
        error = lldb.SBError()
        # Don't read in dependencies so we don't come across false matches that
        # add unwanted breakpoint hits.
        self.target = self.dbg.CreateTarget(exe, None, None, False, error)
        self.assertTrue(self.target, VALID_TARGET)

        cpp_bp = self.target.BreakpointCreateByName(
            "func_from_cpp",
            lldb.eFunctionNameTypeAuto,
            lldb.eLanguageTypeC_plus_plus,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList())
        self.assertTrue(
            cpp_bp.GetNumLocations() == 1,
            "Only one C++ symbol matches")
        self.assertTrue(self.check_location_file(cpp_bp, 0, "b.cpp"))

        no_cpp_bp = self.target.BreakpointCreateByName(
            "func_from_c",
            lldb.eFunctionNameTypeAuto,
            lldb.eLanguageTypeC_plus_plus,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList())
        self.assertTrue(
            no_cpp_bp.GetNumLocations() == 0,
            "And the C one doesn't match")

        c_bp = self.target.BreakpointCreateByName(
            "func_from_c",
            lldb.eFunctionNameTypeAuto,
            lldb.eLanguageTypeC,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList())
        self.assertTrue(
            c_bp.GetNumLocations() == 1,
            "Only one C symbol matches")
        self.assertTrue(self.check_location_file(c_bp, 0, "a.c"))

        no_c_bp = self.target.BreakpointCreateByName(
            "func_from_cpp",
            lldb.eFunctionNameTypeAuto,
            lldb.eLanguageTypeC,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList())
        self.assertTrue(
            no_c_bp.GetNumLocations() == 0,
            "And the C++ one doesn't match")

        objc_bp = self.target.BreakpointCreateByName(
            "func_from_cpp",
            lldb.eFunctionNameTypeAuto,
            lldb.eLanguageTypeObjC,
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList())
        self.assertTrue(
            objc_bp.GetNumLocations() == 0,
            "No ObjC symbol matches")
