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
    NO_DEBUG_INFO_TESTCASE = True

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

    def test_configuring_names(self):
        """Use Python APIs to test that configuring options on breakpoint names works correctly."""
        self.build()
        self.make_a_dummy_name()
        self.setup_target()
        self.do_check_configuring_names()

    def test_configuring_permissions_sb(self):
        """Use Python APIs to test that configuring permissions on names works correctly."""
        self.build()
        self.setup_target()
        self.do_check_configuring_permissions_sb()

    def test_configuring_permissions_cli(self):
        """Use Python APIs to test that configuring permissions on names works correctly."""
        self.build()
        self.setup_target()
        self.do_check_configuring_permissions_cli()

    def setup_target(self):
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a targets we are making breakpoint in and copying to:
        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)
        self.main_file_spec = lldb.SBFileSpec(os.path.join(os.getcwd(), "main.c"))
        
    def check_name_in_target(self, bkpt_name):
        name_list = lldb.SBStringList()
        self.target.GetBreakpointNames(name_list)
        found_it = False
        for name in name_list:
            if name == bkpt_name:
                found_it = True
                break
        self.assertTrue(found_it, "Didn't find the name %s in the target's name list:"%(bkpt_name))
       
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        # These are the settings we're going to be putting into names & breakpoints:
        self.bp_name_string = "ABreakpoint"
        self.is_one_shot = True
        self.ignore_count = 1000
        self.condition = "1 == 2"
        self.auto_continue = True
        self.tid = 0xaaaa
        self.tidx = 10
        self.thread_name = "Fooey"
        self.queue_name = "Blooey"
        self.cmd_list = lldb.SBStringList()
        self.cmd_list.AppendString("frame var")
        self.cmd_list.AppendString("bt")


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

        # Make sure the name is also in the target:
        self.check_name_in_target(bkpt_name)
 
        # Add another name, make sure that works too:
        bkpt.AddName(other_bkpt_name)

        matches = bkpt.MatchesName(bkpt_name)
        self.assertTrue(matches, "Adding a name means we didn't match the name we just set")
        self.check_name_in_target(other_bkpt_name)

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
        bad_names = ["-CantStartWithADash",
                     "1CantStartWithANumber",
                     "^CantStartWithNonAlpha",
                     "CantHave-ADash",
                     "Cant Have Spaces"]
        for bad_name in bad_names:
            success = bkpt.AddName(bad_name)
            self.assertTrue(not success,"We allowed an illegal name: %s"%(bad_name))
            bp_name = lldb.SBBreakpointName(self.target, bad_name)
            self.assertFalse(bp_name.IsValid(), "We made a breakpoint name with an illegal name: %s"%(bad_name));

            retval =lldb.SBCommandReturnObject()
            self.dbg.GetCommandInterpreter().HandleCommand("break set -n whatever -N '%s'"%(bad_name), retval)
            self.assertTrue(not retval.Succeeded(), "break set succeeded with: illegal name: %s"%(bad_name))

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
        self.dbg.GetCommandInterpreter().HandleCommand("break modify --one-shot 1 %s"%(other_bkpt_name), retval)
        self.assertTrue(retval.Succeeded(), "break modify failed with: %s."%(retval.GetError()))
        self.assertTrue(not bkpt.IsOneShot(), "We applied one-shot to the wrong breakpoint.")

    def check_option_values(self, bp_object):
        self.assertEqual(bp_object.IsOneShot(), self.is_one_shot, "IsOneShot")
        self.assertEqual(bp_object.GetIgnoreCount(), self.ignore_count, "IgnoreCount")
        self.assertEqual(bp_object.GetCondition(), self.condition, "Condition")
        self.assertEqual(bp_object.GetAutoContinue(), self.auto_continue, "AutoContinue")
        self.assertEqual(bp_object.GetThreadID(), self.tid, "Thread ID")
        self.assertEqual(bp_object.GetThreadIndex(), self.tidx, "Thread Index")
        self.assertEqual(bp_object.GetThreadName(), self.thread_name, "Thread Name")
        self.assertEqual(bp_object.GetQueueName(), self.queue_name, "Queue Name")
        set_cmds = lldb.SBStringList()
        bp_object.GetCommandLineCommands(set_cmds)
        self.assertEqual(set_cmds.GetSize(), self.cmd_list.GetSize(), "Size of command line commands")
        for idx in range(0, set_cmds.GetSize()):
            self.assertEqual(self.cmd_list.GetStringAtIndex(idx), set_cmds.GetStringAtIndex(idx), "Command %d"%(idx))

    def make_a_dummy_name(self):
        "This makes a breakpoint name in the dummy target to make sure it gets copied over"

        dummy_target = self.dbg.GetDummyTarget()
        self.assertTrue(dummy_target.IsValid(), "Dummy target was not valid.")

        def cleanup ():
            self.dbg.GetDummyTarget().DeleteBreakpointName(self.bp_name_string)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # Now find it in the dummy target, and make sure these settings took:
        bp_name = lldb.SBBreakpointName(dummy_target, self.bp_name_string)
        # Make sure the name is right:
        self.assertTrue (bp_name.GetName() == self.bp_name_string, "Wrong bp_name: %s"%(bp_name.GetName()))
        bp_name.SetOneShot(self.is_one_shot)
        bp_name.SetIgnoreCount(self.ignore_count)
        bp_name.SetCondition(self.condition)
        bp_name.SetAutoContinue(self.auto_continue)
        bp_name.SetThreadID(self.tid)
        bp_name.SetThreadIndex(self.tidx)
        bp_name.SetThreadName(self.thread_name)
        bp_name.SetQueueName(self.queue_name)
        bp_name.SetCommandLineCommands(self.cmd_list)

        # Now look it up again, and make sure it got set correctly.
        bp_name = lldb.SBBreakpointName(dummy_target, self.bp_name_string)
        self.assertTrue(bp_name.IsValid(), "Failed to make breakpoint name.")
        self.check_option_values(bp_name)

    def do_check_configuring_names(self):
        """Use Python APIs to check that configuring breakpoint names works correctly."""
        other_bp_name_string = "AnotherBreakpointName"
        cl_bp_name_string = "CLBreakpointName"

        # Now find the version copied in from the dummy target, and make sure these settings took:
        bp_name = lldb.SBBreakpointName(self.target, self.bp_name_string)
        self.assertTrue(bp_name.IsValid(), "Failed to make breakpoint name.")
        self.check_option_values(bp_name)

        # Now add this name to a breakpoint, and make sure it gets configured properly
        bkpt = self.target.BreakpointCreateByLocation(self.main_file_spec, 10)
        success = bkpt.AddName(self.bp_name_string)
        self.assertTrue(success, "Couldn't add this name to the breakpoint")
        self.check_option_values(bkpt)

        # Now make a name from this breakpoint, and make sure the new name is properly configured:
        new_name = lldb.SBBreakpointName(bkpt, other_bp_name_string)
        self.assertTrue(new_name.IsValid(), "Couldn't make a valid bp_name from a breakpoint.")
        self.check_option_values(bkpt)

        # Now change the name's option and make sure it gets propagated to
        # the breakpoint:
        new_auto_continue = not self.auto_continue
        bp_name.SetAutoContinue(new_auto_continue)
        self.assertEqual(bp_name.GetAutoContinue(), new_auto_continue, "Couldn't change auto-continue on the name")
        self.assertEqual(bkpt.GetAutoContinue(), new_auto_continue, "Option didn't propagate to the breakpoint.")
        
        # Now make this same breakpoint name - but from the command line
        cmd_str = "breakpoint name configure %s -o %d -i %d -c '%s' -G %d -t %d -x %d -T '%s' -q '%s'"%(cl_bp_name_string, 
                                                                             self.is_one_shot, 
                                                                             self.ignore_count, 
                                                                             self.condition, 
                                                                             self.auto_continue,
                                                                             self.tid,
                                                                             self.tidx,
                                                                             self.thread_name,
                                                                             self.queue_name)
        for cmd in self.cmd_list:
            cmd_str += " -C '%s'"%(cmd)
        
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(cmd_str, result)
        self.assertTrue(result.Succeeded())
        # Now look up this name again and check its options:
        cl_name = lldb.SBBreakpointName(self.target, cl_bp_name_string)
        self.check_option_values(cl_name)
        
        # We should have three names now, make sure the target can list them:
        name_list = lldb.SBStringList()
        self.target.GetBreakpointNames(name_list)
        for name_string in [self.bp_name_string, other_bp_name_string, cl_bp_name_string]:
            self.assertTrue(name_string in name_list, "Didn't find %s in names"%(name_string))

        # Test that deleting the name we injected into the dummy target works (there's also a
        # cleanup that will do this, but that won't test the result...
        dummy_target = self.dbg.GetDummyTarget()
        dummy_target.DeleteBreakpointName(self.bp_name_string)
        name_list.Clear()
        dummy_target.GetBreakpointNames(name_list)
        self.assertTrue(self.bp_name_string not in name_list, "Didn't delete %s from the dummy target"%(self.bp_name_string))
        
    def check_permission_results(self, bp_name):
        self.assertEqual(bp_name.GetAllowDelete(), False, "Didn't set allow delete.")
        protected_bkpt = self.target.BreakpointCreateByLocation(self.main_file_spec, 10)
        protected_id = protected_bkpt.GetID()

        unprotected_bkpt = self.target.BreakpointCreateByLocation(self.main_file_spec, 10)
        unprotected_id = unprotected_bkpt.GetID()

        success = protected_bkpt.AddName(self.bp_name_string)
        self.assertTrue(success, "Couldn't add this name to the breakpoint")

        self.target.DisableAllBreakpoints()
        self.assertEqual(protected_bkpt.IsEnabled(), True, "Didnt' keep breakpoint from being disabled")
        self.assertEqual(unprotected_bkpt.IsEnabled(), False, "Protected too many breakpoints from disabling.")

        # Try from the command line too:
        unprotected_bkpt.SetEnabled(True)
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("break disable", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual(protected_bkpt.IsEnabled(), True, "Didnt' keep breakpoint from being disabled")
        self.assertEqual(unprotected_bkpt.IsEnabled(), False, "Protected too many breakpoints from disabling.")

        self.target.DeleteAllBreakpoints()
        bkpt = self.target.FindBreakpointByID(protected_id)
        self.assertTrue(bkpt.IsValid(), "Didn't keep the breakpoint from being deleted.")
        bkpt = self.target.FindBreakpointByID(unprotected_id)
        self.assertFalse(bkpt.IsValid(), "Protected too many breakpoints from deletion.")

        # Remake the unprotected breakpoint and try again from the command line:
        unprotected_bkpt = self.target.BreakpointCreateByLocation(self.main_file_spec, 10)
        unprotected_id = unprotected_bkpt.GetID()

        self.dbg.GetCommandInterpreter().HandleCommand("break delete -f", result)
        self.assertTrue(result.Succeeded())
        bkpt = self.target.FindBreakpointByID(protected_id)
        self.assertTrue(bkpt.IsValid(), "Didn't keep the breakpoint from being deleted.")
        bkpt = self.target.FindBreakpointByID(unprotected_id)
        self.assertFalse(bkpt.IsValid(), "Protected too many breakpoints from deletion.")

    def do_check_configuring_permissions_sb(self):
        bp_name = lldb.SBBreakpointName(self.target, self.bp_name_string)

        # Make a breakpoint name with delete disallowed:
        bp_name = lldb.SBBreakpointName(self.target, self.bp_name_string)
        self.assertTrue(bp_name.IsValid(), "Failed to make breakpoint name for valid name.")

        bp_name.SetAllowDelete(False)
        bp_name.SetAllowDisable(False)
        bp_name.SetAllowList(False)
        self.check_permission_results(bp_name)

    def do_check_configuring_permissions_cli(self):
        # Make the name with the right options using the command line:
        self.runCmd("breakpoint name configure -L 0 -D 0 -A 0 %s"%(self.bp_name_string), check=True)
        # Now look up the breakpoint we made, and check that it works.
        bp_name = lldb.SBBreakpointName(self.target, self.bp_name_string)
        self.assertTrue(bp_name.IsValid(), "Didn't make a breakpoint name we could find.")
        self.check_permission_results(bp_name)
