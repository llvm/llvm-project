"""
Test breakpoint ignore count features.
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointSerialization(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    def test_resolvers(self):
        """Use Python APIs to test that we serialize resolvers."""
        self.build()
        self.setup_targets_and_cleanup()
        self.do_check_resolvers()

    def not_test_filters(self):
        """Use Python APIs to test that we serialize search filters correctly."""
        self.build()
        self.setup_targets_and_cleanup()
        self.check_filters()

    def not_test_options(self):
        """Use Python APIs to test that we serialize breakpoint options correctly."""
        self.build()
        self.setup_targets_and_cleanup()
        self.check_filters()

    def not_test_complex(self):
        """Use Python APIs to test that we serialize complex breakpoints correctly."""
        self.build()
        self.setup_targets_and_cleanup()
        self.check_filters()

    def setup_targets_and_cleanup(self):
        def cleanup ():
            #self.RemoveTempFile(self.bkpts_file_path)

            if self.orig_target.IsValid():
                self.dbg.DeleteTarget(self.orig_target)
                self.dbg.DeleteTarget(self.copy_target)

        self.addTearDownHook(cleanup)
        #self.RemoveTempFile(self.bkpts_file_path)

        exe = os.path.join(os.getcwd(), "a.out")

        # Create a targets we are making breakpoint in and copying to:
        self.orig_target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.orig_target, VALID_TARGET)
        
        self.copy_target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.copy_target, VALID_TARGET)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.bkpts_file_path = os.path.join(os.getcwd(), "breakpoints.json")
        self.bkpts_file_spec = lldb.SBFileSpec(self.bkpts_file_path)

    def do_check_resolvers(self):
        """Use Python APIs to check serialization of breakpoint resolvers"""

        empty_module_list = lldb.SBFileSpecList()
        empty_cu_list = lldb.SBFileSpecList()
        blubby_file_spec = lldb.SBFileSpec(os.path.join(os.getcwd(), "blubby.c"))

        # It isn't actually important for these purposes that these breakpoint
        # actually have locations.
        source_bps = lldb.SBBreakpointList(self.orig_target)
        source_bps.Append(self.orig_target.BreakpointCreateByLocation("blubby.c", 666))
        source_bps.Append(self.orig_target.BreakpointCreateByName("blubby", lldb.eFunctionNameTypeAuto, empty_module_list, empty_cu_list))
        source_bps.Append(self.orig_target.BreakpointCreateByName("blubby", lldb.eFunctionNameTypeFull, empty_module_list,empty_cu_list))
        source_bps.Append(self.orig_target.BreakpointCreateBySourceRegex("dont really care", blubby_file_spec))

        error = lldb.SBError()
        error = self.orig_target.BreakpointsWriteToFile(self.bkpts_file_spec)
        self.assertTrue(error.Success(), "Failed writing breakpoints to file: %s."%(error.GetCString()))

        copy_bps = lldb.SBBreakpointList(self.copy_target)
        error = self.copy_target.BreakpointsCreateFromFile(self.bkpts_file_spec, copy_bps)
        self.assertTrue(error.Success(), "Failed reading breakpoints from file: %s"%(error.GetCString()))

        num_source_bps = source_bps.GetSize()
        num_copy_bps = copy_bps.GetSize()
        self.assertTrue(num_source_bps == num_copy_bps, "Didn't get same number of input and output breakpoints - orig: %d copy: %d"%(num_source_bps, num_copy_bps))
        
        for i in range(0, num_source_bps):
            source_bp = source_bps.GetBreakpointAtIndex(i)
            source_desc = lldb.SBStream()
            source_bp.GetDescription(source_desc, False)
            source_text = source_desc.GetData()

            # I am assuming here that the breakpoints will get written out in breakpoint ID order, and
            # read back in ditto.  That is true right now, and I can't see any reason to do it differently
            # but if we do we can go to writing the breakpoints one by one, or sniffing the descriptions to
            # see which one is which.
            copy_id = source_bp.GetID()
            copy_bp = copy_bps.FindBreakpointByID(copy_id)
            self.assertTrue(copy_bp.IsValid(), "Could not find copy breakpoint %d."%(copy_id))

            copy_desc = lldb.SBStream()
            copy_bp.GetDescription(copy_desc, False)
            copy_text = copy_desc.GetData()

            # These two should be identical.
            print ("Source test for %d is %s."%(i, source_text))
            self.assertTrue (source_text == copy_text, "Source and dest breakpoints are not identical: \nsource: %s\ndest: %s"%(source_text, copy_text))

    def check_filters(self):
        """Use Python APIs to check serialization of breakpoint filters."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

    def check_options(self):
        """Use Python APIs to check serialization of breakpoint options."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

    def check_resolvers(self):
        """Use Python APIs to check serialization of breakpoint resolvers."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)


        
