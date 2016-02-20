"""
Test getting return-values correctly when stepping out
"""

from __future__ import print_function



import os, time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class ReturnValueTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(oslist=["macosx","freebsd"], archs=["i386"])
    @expectedFailureAll(oslist=["linux"], compiler="clang", compiler_version=["<=", "3.6"], archs=["i386"])
    @expectedFailureAll(bugnumber="llvm.org/pr25785", hostoslist=["windows"], compiler="gcc", archs=["i386"], triple='.*-android')
    @expectedFailureWindows("llvm.org/pr24778")
    @add_test_categories(['pyapi'])
    def test_with_python(self):
        """Test getting return values from stepping out."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        error = lldb.SBError()

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        inner_sint_bkpt = self.target.BreakpointCreateByName("inner_sint", exe)
        self.assertTrue(inner_sint_bkpt, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        self.process = self.target.LaunchSimple (None, None, self.get_process_working_directory())

        self.assertTrue(self.process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.assertTrue(self.process.GetState() == lldb.eStateStopped,
                        STOPPED_DUE_TO_BREAKPOINT)

        # Now finish, and make sure the return value is correct.
        thread = lldbutil.get_stopped_thread (self.process, lldb.eStopReasonBreakpoint)

        # inner_sint returns the variable value, so capture that here:
        in_int = thread.GetFrameAtIndex(0).FindVariable ("value").GetValueAsSigned(error)
        self.assertTrue (error.Success())

        thread.StepOut();

        self.assertTrue (self.process.GetState() == lldb.eStateStopped)
        self.assertTrue (thread.GetStopReason() == lldb.eStopReasonPlanComplete)

        frame = thread.GetFrameAtIndex(0)
        fun_name = frame.GetFunctionName()
        self.assertTrue (fun_name == "outer_sint")

        return_value = thread.GetStopReturnValue()
        self.assertTrue (return_value.IsValid())

        ret_int = return_value.GetValueAsSigned(error)
        self.assertTrue (error.Success())
        self.assertTrue (in_int == ret_int)

        # Run again and we will stop in inner_sint the second time outer_sint is called.  
        #Then test stepping out two frames at once:

        self.process.Continue()
        thread_list = lldbutil.get_threads_stopped_at_breakpoint (self.process, inner_sint_bkpt)
        self.assertTrue(len(thread_list) == 1)
        thread = thread_list[0]

        # We are done with the inner_sint breakpoint:
        self.target.BreakpointDelete (inner_sint_bkpt.GetID())

        frame = thread.GetFrameAtIndex(1)
        fun_name = frame.GetFunctionName ()
        self.assertTrue (fun_name == "outer_sint")
        in_int = frame.FindVariable ("value").GetValueAsSigned(error)
        self.assertTrue (error.Success())

        thread.StepOutOfFrame (frame)

        self.assertTrue (self.process.GetState() == lldb.eStateStopped)
        self.assertTrue (thread.GetStopReason() == lldb.eStopReasonPlanComplete)
        frame = thread.GetFrameAtIndex(0)
        fun_name = frame.GetFunctionName()
        self.assertTrue (fun_name == "main")

        ret_value = thread.GetStopReturnValue()
        self.assertTrue (return_value.IsValid())
        ret_int = ret_value.GetValueAsSigned (error)
        self.assertTrue (error.Success())
        self.assertTrue (2 * in_int == ret_int)
        
        # Now try some simple returns that have different types:
        inner_float_bkpt = self.target.BreakpointCreateByName("inner_float", exe)
        self.assertTrue(inner_float_bkpt, VALID_BREAKPOINT)
        self.process.Continue()
        thread_list = lldbutil.get_threads_stopped_at_breakpoint (self.process, inner_float_bkpt)
        self.assertTrue (len(thread_list) == 1)
        thread = thread_list[0]

        self.target.BreakpointDelete (inner_float_bkpt.GetID())

        frame = thread.GetFrameAtIndex(0)
        in_value = frame.FindVariable ("value")
        in_float = float (in_value.GetValue())
        thread.StepOut()

        self.assertTrue (self.process.GetState() == lldb.eStateStopped)
        self.assertTrue (thread.GetStopReason() == lldb.eStopReasonPlanComplete)

        frame = thread.GetFrameAtIndex(0)
        fun_name = frame.GetFunctionName()
        self.assertTrue (fun_name == "outer_float")

        return_value = thread.GetStopReturnValue()
        self.assertTrue (return_value.IsValid())
        return_float = float (return_value.GetValue())

        self.assertTrue(in_float == return_float)

        self.return_and_test_struct_value ("return_one_int")
        self.return_and_test_struct_value ("return_two_int")
        self.return_and_test_struct_value ("return_three_int")
        self.return_and_test_struct_value ("return_four_int")
        self.return_and_test_struct_value ("return_five_int")
        
        self.return_and_test_struct_value ("return_two_double")
        self.return_and_test_struct_value ("return_one_double_two_float")
        self.return_and_test_struct_value ("return_one_int_one_float_one_int")
        
        self.return_and_test_struct_value ("return_one_pointer")
        self.return_and_test_struct_value ("return_two_pointer")
        self.return_and_test_struct_value ("return_one_float_one_pointer")
        self.return_and_test_struct_value ("return_one_int_one_pointer")
        self.return_and_test_struct_value ("return_three_short_one_float")

        self.return_and_test_struct_value ("return_one_int_one_double")
        self.return_and_test_struct_value ("return_one_int_one_double_one_int")
        self.return_and_test_struct_value ("return_one_short_one_double_one_short")
        self.return_and_test_struct_value ("return_one_float_one_int_one_float")
        self.return_and_test_struct_value ("return_two_float")
        # I am leaving out the packed test until we have a way to tell CLANG 
        # about alignment when reading DWARF for packed types.
        #self.return_and_test_struct_value ("return_one_int_one_double_packed")
        self.return_and_test_struct_value ("return_one_int_one_long")

        # icc and gcc don't support this extension.
        if self.getCompiler().endswith('clang'):
            self.return_and_test_struct_value ("return_vector_size_float32_8")
            self.return_and_test_struct_value ("return_vector_size_float32_16")
            self.return_and_test_struct_value ("return_vector_size_float32_32")
            self.return_and_test_struct_value ("return_ext_vector_size_float32_2")
            self.return_and_test_struct_value ("return_ext_vector_size_float32_4")
            self.return_and_test_struct_value ("return_ext_vector_size_float32_8")

    def return_and_test_struct_value (self, func_name):
        """Pass in the name of the function to return from - takes in value, returns value."""
        
        # Set the breakpoint, run to it, finish out.
        bkpt = self.target.BreakpointCreateByName (func_name)
        self.assertTrue (bkpt.GetNumResolvedLocations() > 0)

        self.process.Continue ()

        thread_list = lldbutil.get_threads_stopped_at_breakpoint (self.process, bkpt)

        self.assertTrue (len(thread_list) == 1)
        thread = thread_list[0]

        self.target.BreakpointDelete (bkpt.GetID())

        in_value = thread.GetFrameAtIndex(0).FindVariable ("value")
        
        self.assertTrue (in_value.IsValid())
        num_in_children = in_value.GetNumChildren()

        # This is a little hokey, but if we don't get all the children now, then
        # once we've stepped we won't be able to get them?
        
        for idx in range(0, num_in_children):
            in_child = in_value.GetChildAtIndex (idx)
            in_child_str = in_child.GetValue()

        thread.StepOut()
        
        self.assertTrue (self.process.GetState() == lldb.eStateStopped)
        self.assertTrue (thread.GetStopReason() == lldb.eStopReasonPlanComplete)

        # Assuming all these functions step out to main.  Could figure out the caller dynamically
        # if that would add something to the test.
        frame = thread.GetFrameAtIndex(0)
        fun_name = frame.GetFunctionName()
        self.assertTrue (fun_name == "main")

        frame = thread.GetFrameAtIndex(0)
        ret_value = thread.GetStopReturnValue()

        self.assertTrue (ret_value.IsValid())

        num_ret_children = ret_value.GetNumChildren()
        self.assertTrue (num_in_children == num_ret_children)
        for idx in range(0, num_ret_children):
            in_child = in_value.GetChildAtIndex(idx)
            ret_child = ret_value.GetChildAtIndex(idx)
            in_child_str = in_child.GetValue()
            ret_child_str = ret_child.GetValue()

            self.assertEqual(in_child_str, ret_child_str)
