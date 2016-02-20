"""
Test SBValue API linked_list_iter which treats the SBValue as a linked list and
supports iteration till the end of list is reached.
"""

from __future__ import print_function



import os, time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class ValueAsLinkedListTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Break at this line')

    # Py3 asserts due to a bug in SWIG.  A fix for this was upstreamed into SWIG 3.0.8.
    @skipIf(py_version=['>=', (3,0)], swig_version=['<', (3,0,8)])
    @add_test_categories(['pyapi'])
    def test(self):
        """Exercise SBValue API linked_list_iter."""
        d = {'EXE': self.exe_name}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        exe = os.path.join(os.getcwd(), self.exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation('main.cpp', self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Get Frame #0.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)

        # Get variable 'task_head'.
        task_head = frame0.FindVariable('task_head')
        self.assertTrue(task_head, VALID_VARIABLE)
        self.DebugSBValue(task_head)

        # By design (see main.cpp), the visited id's are: [1, 2, 4, 5].
        visitedIDs = [1, 2, 4, 5]
        list = []

        cvf = lldbutil.ChildVisitingFormatter(indent_child=2)
        for t in task_head.linked_list_iter('next'):
            self.assertTrue(t, VALID_VARIABLE)
            # Make sure that 'next' corresponds to an SBValue with pointer type.
            self.assertTrue(t.TypeIsPointerType())
            if self.TraceOn():
                print(cvf.format(t))
            list.append(int(t.GetChildMemberWithName("id").GetValue()))

        # Sanity checks that the we visited all the items (no more, no less).
        if self.TraceOn():
            print("visited IDs:", list)
        self.assertTrue(visitedIDs == list)

        # Let's exercise the linked_list_iter() API again, this time supplying
        # our end of list test function.
        def eol(val):
            """Test function to determine end of list."""
            # End of list is reached if either the value object is invalid
            # or it corresponds to a null pointer.
            if not val or int(val.GetValue(), 16) == 0:
                return True
            # Also check the "id" for correct semantics.  If id <= 0, the item
            # is corrupted, let's return True to signify end of list.
            if int(val.GetChildMemberWithName("id").GetValue(), 0) <= 0:
                return True

            # Otherwise, return False.
            return False

        list = []
        for t in task_head.linked_list_iter('next', eol):
            self.assertTrue(t, VALID_VARIABLE)
            # Make sure that 'next' corresponds to an SBValue with pointer type.
            self.assertTrue(t.TypeIsPointerType())
            if self.TraceOn():
                print(cvf.format(t))
            list.append(int(t.GetChildMemberWithName("id").GetValue()))

        # Sanity checks that the we visited all the items (no more, no less).
        if self.TraceOn():
            print("visited IDs:", list)
        self.assertTrue(visitedIDs == list)
        
        # Get variable 'empty_task_head'.
        empty_task_head = frame0.FindVariable('empty_task_head')
        self.assertTrue(empty_task_head, VALID_VARIABLE)
        self.DebugSBValue(empty_task_head)

        list = []
        # There is no iterable item from empty_task_head.linked_list_iter().
        for t in empty_task_head.linked_list_iter('next', eol):
            if self.TraceOn():
                print(cvf.format(t))
            list.append(int(t.GetChildMemberWithName("id").GetValue()))

        self.assertTrue(len(list) == 0)

        # Get variable 'task_evil'.
        task_evil = frame0.FindVariable('task_evil')
        self.assertTrue(task_evil, VALID_VARIABLE)
        self.DebugSBValue(task_evil)

        list = []
        # There 3 iterable items from task_evil.linked_list_iter(). :-)
        for t in task_evil.linked_list_iter('next'):
            if self.TraceOn():
                print(cvf.format(t))
            list.append(int(t.GetChildMemberWithName("id").GetValue()))

        self.assertTrue(len(list) == 3)
