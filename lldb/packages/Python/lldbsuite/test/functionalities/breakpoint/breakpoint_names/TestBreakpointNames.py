"""
Test breakpoint names.
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointNames(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    def test_setting_names(self):
        """Use Python APIs to test that we can set breakpoint names."""
        self.build()
        self.setup_target()
        self.do_check_names()

    def test_illegal_names(self):
        """Use Python APIs to test that we don't allow illegal names."""
        self.build()
        self.setup_target()
        self.do_check_illegal_names()

    def test_using_names(self):
        """Use Python APIs to test that operations on names works correctly."""
        self.build()
        self.setup_target()
        self.do_check_using_names()

    def setup_target(self):
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a targets we are making breakpoint in and copying to:
        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)
        self.main_file_spec = lldb.SBFileSpec(os.path.join(os.getcwd(), "main.c"))
        
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def do_check_names(self):
        """Use Python APIs to check that we can set & retrieve breakpoint names"""
        bkpt = self.target.BreakpointCreateByLocation(self.main_file_spec, 10)
        bkpt_name = "ABreakpoint"
        other_bkpt_name = "_AnotherBreakpoint"

        # Add a name and make sure we match it:
        success = bkpt.AddName(bkpt_name)
        self.assertTrue(success, "We couldn't add a legal name to a breakpoint.")

        matches = bkpt.MatchesName(bkpt_name)
        self.assertTrue(matches, "We didn't match the name we just set")
        
        # Make sure we don't match irrelevant names:
        matches = bkpt.MatchesName("NotABreakpoint")
        self.assertTrue(not matches, "We matched a name we didn't set.")

        # Add another name, make sure that works too:
        bkpt.AddName(other_bkpt_name)

        matches = bkpt.MatchesName(bkpt_name)
        self.assertTrue(matches, "Adding a name means we didn't match the name we just set")

        # Remove the name and make sure we no longer match it:
        bkpt.RemoveName(bkpt_name)
        matches = bkpt.MatchesName(bkpt_name)
        self.assertTrue(not matches,"We still match a name after removing it.")

        # Make sure the name list has the remaining name:
        name_list = lldb.SBStringList()
        bkpt.GetNames(name_list)
        num_names = name_list.GetSize()
        self.assertTrue(num_names == 1, "Name list has %d items, expected 1."%(num_names))
        
        name = name_list.GetStringAtIndex(0)
        self.assertTrue(name == other_bkpt_name, "Remaining name was: %s expected %s."%(name, other_bkpt_name))

    def do_check_illegal_names(self):
        """Use Python APIs to check that we reject illegal names."""
        bkpt = self.target.BreakpointCreateByLocation(self.main_file_spec, 10)
        success = bkpt.AddName("-CantStartWithADash")
        self.assertTrue(not success,"We allowed a name starting with a dash.")

        success = bkpt.AddName("1CantStartWithANumber")
        self.assertTrue(not success, "We allowed a name starting with a number.")

        success = bkpt.AddName("^CantStartWithNonAlpha")
        self.assertTrue(not success, "We allowed a name starting with an ^.")

        success = bkpt.AddName("CantHave-ADash")
        self.assertTrue(not success, "We allowed a name with a dash in it.")

        success = bkpt.AddName("Cant Have Spaces")
        self.assertTrue(not success, "We allowed a name with spaces.")

        # Check from the command line as well:
        retval =lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("break set -n whatever -N has-dashes", retval)
        self.assertTrue(not retval.Succeeded(), "break set succeeded with: illegal name")
        
    def do_check_using_names(self):
        """Use Python APIs to check names work in place of breakpoint ID's."""
        
        bkpt = self.target.BreakpointCreateByLocation(self.main_file_spec, 10)
        bkpt_name = "ABreakpoint"
        other_bkpt_name= "_AnotherBreakpoint"

        # Add a name and make sure we match it:
        success = bkpt.AddName(bkpt_name)
        self.assertTrue(success, "We couldn't add a legal name to a breakpoint.")

        bkpts = lldb.SBBreakpointList(self.target)
        self.target.FindBreakpointsByName(bkpt_name, bkpts)

        self.assertTrue(bkpts.GetSize() == 1, "One breakpoint matched.")
        found_bkpt = bkpts.GetBreakpointAtIndex(0)
        self.assertTrue(bkpt.GetID() == found_bkpt.GetID(),"The right breakpoint.")

        retval = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("break disable %s"%(bkpt_name), retval)
        self.assertTrue(retval.Succeeded(), "break disable failed with: %s."%(retval.GetError()))
        self.assertTrue(not bkpt.IsEnabled(), "We didn't disable the breakpoint.")

        # Also make sure we don't apply commands to non-matching names:
        self.dbg.GetCommandInterpreter().HandleCommand("break modify --one-shot 1 %s"%(bkpt_name), retval)
        self.assertTrue(retval.Succeeded(), "break modify failed with: %s."%(retval.GetError()))
        self.assertTrue(not bkpt.IsOneShot(), "We applied one-shot to the wrong breakpoint.")

        
        
