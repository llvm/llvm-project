"""
Use lldb Python SBFrame API to get the argument values of the call stacks.
And other SBFrame API tests.
"""

from __future__ import print_function



import os, time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class FrameAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @expectedFailureWindows("llvm.org/pr24778")
    def test_get_arg_vals_for_call_stack(self):
        """Exercise SBFrame.GetVariables() API to get argument vals."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        #print("breakpoint:", breakpoint)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        process = target.GetProcess()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        # Keeps track of the number of times 'a' is called where it is within a
        # depth of 3 of the 'c' leaf function.
        callsOfA = 0

        from six import StringIO as SixStringIO
        session = SixStringIO()
        while process.GetState() == lldb.eStateStopped:
            thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
            self.assertIsNotNone(thread)
            # Inspect at most 3 frames.
            numFrames = min(3, thread.GetNumFrames())
            for i in range(numFrames):
                frame = thread.GetFrameAtIndex(i)
                if self.TraceOn():
                    print("frame:", frame)

                name = frame.GetFunction().GetName()
                if name == 'a':
                    callsOfA = callsOfA + 1

                # We'll inspect only the arguments for the current frame:
                #
                # arguments     => True
                # locals        => False
                # statics       => False
                # in_scope_only => True
                valList = frame.GetVariables(True, False, False, True)
                argList = []
                for val in valList:
                    argList.append("(%s)%s=%s" % (val.GetTypeName(),
                                                  val.GetName(),
                                                  val.GetValue()))
                print("%s(%s)" % (name, ", ".join(argList)), file=session)
                
                # Also check the generic pc & stack pointer.  We can't test their absolute values,
                # but they should be valid.  Uses get_GPRs() from the lldbutil module.
                gpr_reg_set = lldbutil.get_GPRs(frame)
                pc_value = gpr_reg_set.GetChildMemberWithName("pc")
                self.assertTrue (pc_value, "We should have a valid PC.")
                pc_value_int = int(pc_value.GetValue(), 0)
                # Make sure on arm targets we dont mismatch PC value on the basis of thumb bit.
                # Frame PC will not have thumb bit set in case of a thumb instruction as PC.
                if self.getArchitecture() in ['arm']:
                    pc_value_int &= ~1
                self.assertTrue (pc_value_int == frame.GetPC(), "PC gotten as a value should equal frame's GetPC")
                sp_value = gpr_reg_set.GetChildMemberWithName("sp")
                self.assertTrue (sp_value, "We should have a valid Stack Pointer.")
                self.assertTrue (int(sp_value.GetValue(), 0) == frame.GetSP(), "SP gotten as a value should equal frame's GetSP")

            print("---", file=session)
            process.Continue()

        # At this point, the inferior process should have exited.
        self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)

        # Expect to find 'a' on the call stacks two times.
        self.assertTrue(callsOfA == 2,
                        "Expect to find 'a' on the call stacks two times")
        # By design, the 'a' call frame has the following arg vals:
        #     o a((int)val=1, (char)ch='A')
        #     o a((int)val=3, (char)ch='A')
        if self.TraceOn():
            print("Full stack traces when stopped on the breakpoint 'c':")
            print(session.getvalue())
        self.expect(session.getvalue(), "Argugment values displayed correctly",
                    exe=False,
            substrs = ["a((int)val=1, (char)ch='A')",
                       "a((int)val=3, (char)ch='A')"])

    @add_test_categories(['pyapi'])
    def test_frame_api_boundary_condition(self):
        """Exercise SBFrame APIs with boundary condition inputs."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        #print("breakpoint:", breakpoint)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        process = target.GetProcess()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)
        frame = thread.GetFrameAtIndex(0)
        if self.TraceOn():
            print("frame:", frame)

        # Boundary condition testings.
        val1 = frame.FindVariable(None, True)
        val2 = frame.FindVariable(None, False)
        val3 = frame.FindValue(None, lldb.eValueTypeVariableGlobal)
        if self.TraceOn():
            print("val1:", val1)
            print("val2:", val2)

        frame.EvaluateExpression(None)

    @add_test_categories(['pyapi'])
    def test_frame_api_IsEqual(self):
        """Exercise SBFrame API IsEqual."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        #print("breakpoint:", breakpoint)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        process = target.GetProcess()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        frameEntered = thread.GetFrameAtIndex(0)
        if self.TraceOn():
            print(frameEntered)
            lldbutil.print_stacktrace(thread)
        self.assertTrue(frameEntered)

        # Doing two step overs while still inside c().
        thread.StepOver()
        thread.StepOver()
        self.assertTrue(thread)
        frameNow = thread.GetFrameAtIndex(0)
        if self.TraceOn():
            print(frameNow)
            lldbutil.print_stacktrace(thread)
        self.assertTrue(frameNow)

        # The latest two frames are considered equal.
        self.assertTrue(frameEntered.IsEqual(frameNow))

        # Now let's step out of frame c().
        thread.StepOutOfFrame(frameNow)
        frameOutOfC = thread.GetFrameAtIndex(0)
        if self.TraceOn():
            print(frameOutOfC)
            lldbutil.print_stacktrace(thread)
        self.assertTrue(frameOutOfC)

        # The latest two frames should not be equal.
        self.assertFalse(frameOutOfC.IsEqual(frameNow))
