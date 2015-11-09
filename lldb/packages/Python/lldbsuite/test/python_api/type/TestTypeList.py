"""
Test SBType and SBTypeList API.
"""

from __future__ import print_function



import os, time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class TypeAndTypeListTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to break at.
        self.source = 'main.cpp'
        self.line = line_number(self.source, '// Break at this line')

    @add_test_categories(['pyapi'])
    def test(self):
        """Exercise SBType and SBTypeList API."""
        d = {'EXE': self.exe_name}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        exe = os.path.join(os.getcwd(), self.exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Get Frame #0.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)

        # Get the type 'Task'.
        type_list = target.FindTypes('Task')
        if self.TraceOn():
            print("Size of type_list from target.FindTypes('Task') query: %d" % type_list.GetSize())
        self.assertTrue(len(type_list) >= 1) # a second Task make be scared up by the Objective-C runtime
        for type in type_list:
            self.assertTrue(type)
            self.DebugSBType(type)
            self.assertFalse(type.IsAnonymousType(), "Task is not anonymous")
            for field in type.fields:
                if field.name == "type":
                    for enum_member in field.type.enum_members:
                        self.assertTrue(enum_member)
                        self.DebugSBType(enum_member.type)
                elif field.name == "my_type_is_nameless":
                    self.assertTrue(field.type.IsAnonymousType(), "my_type_is_nameless has an anonymous type")
                elif field.name == "my_type_is_named":
                    self.assertFalse(field.type.IsAnonymousType(), "my_type_is_named has a named type")

        # Pass an empty string.  LLDB should not crash. :-)
        fuzz_types = target.FindTypes(None)
        fuzz_type = target.FindFirstType(None)

        # Now use the SBTarget.FindFirstType() API to find 'Task'.
        task_type = target.FindFirstType('Task')
        self.assertTrue(task_type)
        self.DebugSBType(task_type)

        # Get the reference type of 'Task', just for fun.
        task_ref_type = task_type.GetReferenceType()
        self.assertTrue(task_ref_type)
        self.DebugSBType(task_ref_type)

        # Get the pointer type of 'Task', which is the same as task_head's type.
        task_pointer_type = task_type.GetPointerType()
        self.assertTrue(task_pointer_type)
        self.DebugSBType(task_pointer_type)

        # Get variable 'task_head'.
        task_head = frame0.FindVariable('task_head')
        self.assertTrue(task_head, VALID_VARIABLE)
        self.DebugSBValue(task_head)
        task_head_type = task_head.GetType()
        self.DebugSBType(task_head_type)
        self.assertTrue(task_head_type.IsPointerType())

        self.assertTrue(task_head_type == task_pointer_type)

        # Get the pointee type of 'task_head'.
        task_head_pointee_type = task_head_type.GetPointeeType()
        self.DebugSBType(task_head_pointee_type)

        self.assertTrue(task_type == task_head_pointee_type)

        # We'll now get the child member 'id' from 'task_head'.
        id = task_head.GetChildMemberWithName('id')
        self.DebugSBValue(id)
        id_type = id.GetType()
        self.DebugSBType(id_type)

        # SBType.GetBasicType() takes an enum 'BasicType' (lldb-enumerations.h).
        int_type = id_type.GetBasicType(lldb.eBasicTypeInt)
        self.assertTrue(id_type == int_type)
