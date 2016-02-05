"""
Check that types only get completed when necessary.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TypeCompletionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureIcc # often fails with 'NameAndAddress should be valid'
    # Fails with gcc 4.8.1 with llvm.org/pr15301 LLDB prints incorrect sizes of STL containers
    def test_with_run_command(self):
        """Check that types only get completed when necessary."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp (self, "// Set break point at this line.")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type category enable -l c++', check=False)

        self.runCmd('type category disable -l c++', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertFalse(p_type.IsTypeComplete(), 'vector<T> complete but it should not be')

        self.runCmd("continue")

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertFalse(p_type.IsTypeComplete(), 'vector<T> complete but it should not be')

        self.runCmd("continue")

        self.runCmd("frame variable p --show-types")

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertTrue(p_type.IsTypeComplete(), 'vector<T> should now be complete')
        name_address_type = p_type.GetTemplateArgumentType(0)
        self.assertTrue(name_address_type.IsValid(), 'NameAndAddress should be valid')
        self.assertFalse(name_address_type.IsTypeComplete(), 'NameAndAddress complete but it should not be')

        self.runCmd("continue")

        self.runCmd("frame variable guy --show-types")

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertTrue(p_type.IsTypeComplete(), 'vector<T> should now be complete')
        name_address_type = p_type.GetTemplateArgumentType(0)
        self.assertTrue(name_address_type.IsValid(), 'NameAndAddress should be valid')
        self.assertTrue(name_address_type.IsTypeComplete(), 'NameAndAddress should now be complete')
        field0 = name_address_type.GetFieldAtIndex(0)
        self.assertTrue(field0.IsValid(), 'NameAndAddress::m_name should be valid')
        string = field0.GetType().GetPointeeType()
        self.assertTrue(string.IsValid(), 'CustomString should be valid')
        self.assertFalse(string.IsTypeComplete(), 'CustomString complete but it should not be')

        self.runCmd("continue")

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertTrue(p_type.IsTypeComplete(), 'vector<T> should now be complete')
        name_address_type = p_type.GetTemplateArgumentType(0)
        self.assertTrue(name_address_type.IsValid(), 'NameAndAddress should be valid')
        self.assertTrue(name_address_type.IsTypeComplete(), 'NameAndAddress should now be complete')
        field0 = name_address_type.GetFieldAtIndex(0)
        self.assertTrue(field0.IsValid(), 'NameAndAddress::m_name should be valid')
        string = field0.GetType().GetPointeeType()
        self.assertTrue(string.IsValid(), 'CustomString should be valid')
        self.assertFalse(string.IsTypeComplete(), 'CustomString complete but it should not be')

        self.runCmd('type category enable -l c++', check=False)
        self.runCmd('frame variable guy --show-types --ptr-depth=1')

        p_vector = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('p')
        p_type = p_vector.GetType()
        self.assertTrue(p_type.IsTypeComplete(), 'vector<T> should now be complete')
        name_address_type = p_type.GetTemplateArgumentType(0)
        self.assertTrue(name_address_type.IsValid(), 'NameAndAddress should be valid')
        self.assertTrue(name_address_type.IsTypeComplete(), 'NameAndAddress should now be complete')
        field0 = name_address_type.GetFieldAtIndex(0)
        self.assertTrue(field0.IsValid(), 'NameAndAddress::m_name should be valid')
        string = field0.GetType().GetPointeeType()
        self.assertTrue(string.IsValid(), 'CustomString should be valid')
        self.assertTrue(string.IsTypeComplete(), 'CustomString should now be complete')
