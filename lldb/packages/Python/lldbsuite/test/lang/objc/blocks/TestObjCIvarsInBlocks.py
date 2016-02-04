"""Test printing ivars and ObjC objects captured in blocks that are made in methods of an ObjC class."""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestObjCIvarsInBlocks(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "main.m"
        self.class_source = "ivars-in-blocks.m"
        self.class_source_file_spec = lldb.SBFileSpec(self.class_source)

    @skipUnlessDarwin
    @add_test_categories(['pyapi'])
    @expectedFailurei386 # This test requires the 2.0 runtime, so it will fail on i386.
    def test_with_python_api(self):
        """Test printing the ivars of the self when captured in blocks"""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateBySourceRegex ('// Break here inside the block.', self.class_source_file_spec)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        breakpoint_two = target.BreakpointCreateBySourceRegex ('// Break here inside the class method block.', self.class_source_file_spec)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue (process, "Created a process.")
        self.assertTrue (process.GetState() == lldb.eStateStopped, "Stopped it too.")

        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)
        self.assertTrue (len(thread_list) == 1)
        thread = thread_list[0]
        
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue (frame, "frame 0 is valid")
        
        # First use the FindVariable API to see if we can find the ivar by undecorated name:
        direct_blocky = frame.GetValueForVariablePath ("blocky_ivar")
        self.assertTrue(direct_blocky, "Found direct access to blocky_ivar.")
        
        # Now get it as a member of "self" and make sure the two values are equal:
        self_var = frame.GetValueForVariablePath ("self")
        self.assertTrue (self_var, "Found self in block.")
        indirect_blocky = self_var.GetChildMemberWithName ("blocky_ivar")
        self.assertTrue (indirect_blocky, "Found blocky_ivar through self")
        
        error = lldb.SBError()
        direct_value = direct_blocky.GetValueAsSigned(error)
        self.assertTrue (error.Success(), "Got direct value for blocky_ivar")

        indirect_value = indirect_blocky.GetValueAsSigned (error)
        self.assertTrue (error.Success(), "Got indirect value for blocky_ivar")
        
        self.assertTrue (direct_value == indirect_value, "Direct and indirect values are equal.")

        # Now make sure that we can get at the captured ivar through the expression parser.
        # Doing a little trivial math will force this into the real expression parser:
        direct_expr = frame.EvaluateExpression ("blocky_ivar + 10")
        self.assertTrue (direct_expr, "Got blocky_ivar through the expression parser")
        
        # Again, get the value through self directly and make sure they are the same:
        indirect_expr = frame.EvaluateExpression ("self->blocky_ivar + 10")
        self.assertTrue (indirect_expr, "Got blocky ivar through expression parser using self.")
        
        direct_value = direct_expr.GetValueAsSigned (error)
        self.assertTrue (error.Success(), "Got value from direct use of expression parser")

        indirect_value = indirect_expr.GetValueAsSigned (error)
        self.assertTrue (error.Success(), "Got value from indirect access using the expression parser")

        self.assertTrue (direct_value == indirect_value, "Direct ivar access and indirect through expression parser produce same value.")

        process.Continue()
        self.assertTrue (process.GetState() == lldb.eStateStopped, "Stopped at the second breakpoint.")

        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint_two)
        self.assertTrue (len(thread_list) == 1)
        thread = thread_list[0]
        
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue (frame, "frame 0 is valid")
        
        expr = frame.EvaluateExpression("(ret)")
        self.assertTrue (expr, "Successfully got a local variable in a block in a class method.")

        ret_value_signed = expr.GetValueAsSigned (error)
        # print('ret_value_signed = %i' % (ret_value_signed))
        self.assertTrue (ret_value_signed == 5, "The local variable in the block was what we expected.")
