"""Test Python APIs for working with formatters"""

from __future__ import print_function



import os, sys, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class SBFormattersAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.line = line_number('main.cpp', '// Set break point at this line.')

    @add_test_categories(['pyapi'])
    def test_formatters_api(self):
        """Test Python APIs for working with formatters"""
        self.build()
        self.setTearDownCleanup()
        
        """Test Python APIs for working with formatters"""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])
        
        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synthetic clear', check=False)
            self.runCmd('type category delete foobar', check=False)
            self.runCmd('type category delete JASSynth', check=False)
            self.runCmd('type category delete newbar', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)


        format = lldb.SBTypeFormat(lldb.eFormatHex)
        category = self.dbg.GetDefaultCategory()
        category.AddTypeFormat(lldb.SBTypeNameSpecifier("int"),format)

        self.expect("frame variable foo.A",
             substrs = ['0x00000001'])
        self.expect("frame variable foo.E", matching=False,
             substrs = ['b8cca70a'])

        category.AddTypeFormat(lldb.SBTypeNameSpecifier("long"),format)
        self.expect("frame variable foo.A",
             substrs = ['0x00000001'])
        self.expect("frame variable foo.E",
             substrs = ['b8cca70a'])
        
        format.format = lldb.eFormatOctal
        category.AddTypeFormat(lldb.SBTypeNameSpecifier("int"),format)
        self.expect("frame variable foo.A",
             substrs = ['01'])
        self.expect("frame variable foo.E",
             substrs = ['b8cca70a'])
        
        category.DeleteTypeFormat(lldb.SBTypeNameSpecifier("int"))
        category.DeleteTypeFormat(lldb.SBTypeNameSpecifier("long"))
        self.expect("frame variable foo.A", matching=False,
             substrs = ['01'])
        self.expect("frame variable foo.E", matching=False,
             substrs = ['b8cca70a'])

        summary = lldb.SBTypeSummary.CreateWithSummaryString("the hello world you'll never see")
        summary.SetSummaryString('hello world')
        new_category = self.dbg.GetCategory("foobar")
        self.assertFalse(new_category.IsValid(), "getting a non-existing category worked")
        new_category = self.dbg.CreateCategory("foobar")
        new_category.enabled = True
        new_category.AddTypeSummary(lldb.SBTypeNameSpecifier("^.*t$",True),summary)
        self.expect("frame variable foo.A",
             substrs = ['hello world'])
        self.expect("frame variable foo.E", matching=False,
             substrs = ['hello world'])
        self.expect("frame variable foo.B",
             substrs = ['hello world'])
        self.expect("frame variable foo.F",
             substrs = ['hello world'])
        new_category.enabled = False
        self.expect("frame variable foo.A", matching=False,
             substrs = ['hello world'])
        self.expect("frame variable foo.E", matching=False,
             substrs = ['hello world'])
        self.expect("frame variable foo.B", matching=False,
             substrs = ['hello world'])
        self.expect("frame variable foo.F", matching=False,
             substrs = ['hello world'])
        self.dbg.DeleteCategory(new_category.GetName())
        self.expect("frame variable foo.A", matching=False,
             substrs = ['hello world'])
        self.expect("frame variable foo.E", matching=False,
             substrs = ['hello world'])
        self.expect("frame variable foo.B", matching=False,
             substrs = ['hello world'])
        self.expect("frame variable foo.F", matching=False,
             substrs = ['hello world'])

        filter = lldb.SBTypeFilter(0)
        filter.AppendExpressionPath("A")
        filter.AppendExpressionPath("D")
        self.assertTrue(filter.GetNumberOfExpressionPaths() == 2, "filter with two items does not have two items")

        category.AddTypeFilter(lldb.SBTypeNameSpecifier("JustAStruct"),filter)
        self.expect("frame variable foo",
             substrs = ['A = 1', 'D = 6.28'])
        self.expect("frame variable foo", matching=False,
             substrs = ['B = ', 'C = ', 'E = ', 'F = '])

        category.DeleteTypeFilter(lldb.SBTypeNameSpecifier("JustAStruct",True))
        self.expect("frame variable foo",
             substrs = ['A = 1', 'D = 6.28'])
        self.expect("frame variable foo", matching=False,
             substrs = ['B = ', 'C = ', 'E = ', 'F = '])

        category.DeleteTypeFilter(lldb.SBTypeNameSpecifier("JustAStruct",False))
        self.expect("frame variable foo",
             substrs = ['A = 1', 'D = 6.28'])
        self.expect("frame variable foo", matching=True,
             substrs = ['B = ', 'C = ', 'E = ', 'F = '])

        self.runCmd("command script import --allow-reload ./synth.py")

        self.expect("frame variable foo", matching=False,
             substrs = ['X = 1'])

        self.dbg.GetCategory("JASSynth").SetEnabled(True)
        self.expect("frame variable foo", matching=True,
             substrs = ['X = 1'])

        self.dbg.GetCategory("CCCSynth").SetEnabled(True)
        self.expect("frame variable ccc", matching=True,
             substrs = ['CCC object with leading value (int) a = 111', 'a = 111', 'b = 222', 'c = 333'])

        foo_var = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('foo')
        self.assertTrue(foo_var.IsValid(), 'could not find foo')
        self.assertTrue(foo_var.GetDeclaration().IsValid(), 'foo declaration is invalid')

        self.assertTrue(foo_var.GetNumChildren() == 2, 'synthetic value has wrong number of child items (synth)')
        self.assertTrue(foo_var.GetChildMemberWithName('X').GetValueAsUnsigned() == 1, 'foo_synth.X has wrong value (synth)')
        self.assertFalse(foo_var.GetChildMemberWithName('B').IsValid(), 'foo_synth.B is valid but should not (synth)')

        self.dbg.GetCategory("JASSynth").SetEnabled(False)
        foo_var = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('foo')
        self.assertTrue(foo_var.IsValid(), 'could not find foo')

        self.assertFalse(foo_var.GetNumChildren() == 2, 'still seeing synthetic value')

        filter = lldb.SBTypeFilter(0)
        filter.AppendExpressionPath("A")
        filter.AppendExpressionPath("D")
        category.AddTypeFilter(lldb.SBTypeNameSpecifier("JustAStruct"),filter)
        self.expect("frame variable foo",
             substrs = ['A = 1', 'D = 6.28'])

        foo_var = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('foo')
        self.assertTrue(foo_var.IsValid(), 'could not find foo')

        self.assertTrue(foo_var.GetNumChildren() == 2, 'synthetic value has wrong number of child items (filter)')
        self.assertTrue(foo_var.GetChildMemberWithName('X').GetValueAsUnsigned() == 0, 'foo_synth.X has wrong value (filter)')
        self.assertTrue(foo_var.GetChildMemberWithName('A').GetValueAsUnsigned() == 1, 'foo_synth.A has wrong value (filter)')

        self.assertTrue(filter.ReplaceExpressionPathAtIndex(0,"C"), "failed to replace an expression path in filter")
        self.expect("frame variable foo",
             substrs = ['A = 1', 'D = 6.28'])
        category.AddTypeFilter(lldb.SBTypeNameSpecifier("JustAStruct"),filter)
        self.expect("frame variable foo",
             substrs = ["C = 'e'", 'D = 6.28'])
        category.AddTypeFilter(lldb.SBTypeNameSpecifier("FooType"),filter)
        filter.ReplaceExpressionPathAtIndex(1,"F")
        self.expect("frame variable foo",
             substrs = ["C = 'e'", 'D = 6.28'])
        category.AddTypeFilter(lldb.SBTypeNameSpecifier("JustAStruct"),filter)
        self.expect("frame variable foo",
             substrs = ["C = 'e'", 'F = 0'])
        self.expect("frame variable bar",
             substrs = ["C = 'e'", 'D = 6.28'])

        foo_var = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable('foo')
        self.assertTrue(foo_var.IsValid(), 'could not find foo')
        self.assertTrue(foo_var.GetChildMemberWithName('C').GetValueAsUnsigned() == ord('e'), 'foo_synth.C has wrong value (filter)')

        chosen = self.dbg.GetFilterForType(lldb.SBTypeNameSpecifier("JustAStruct"))
        self.assertTrue(chosen.count == 2, "wrong filter found for JustAStruct")
        self.assertTrue(chosen.GetExpressionPathAtIndex(0) == 'C', "wrong item at index 0 for JustAStruct")
        self.assertTrue(chosen.GetExpressionPathAtIndex(1) == 'F', "wrong item at index 1 for JustAStruct")

        self.assertFalse(category.DeleteTypeFilter(lldb.SBTypeNameSpecifier("NoSuchType")),"deleting a non-existing filter worked")
        self.assertFalse(category.DeleteTypeSummary(lldb.SBTypeNameSpecifier("NoSuchType")),"deleting a non-existing summary worked")
        self.assertFalse(category.DeleteTypeFormat(lldb.SBTypeNameSpecifier("NoSuchType")),"deleting a non-existing format worked")
        self.assertFalse(category.DeleteTypeSynthetic(lldb.SBTypeNameSpecifier("NoSuchType")),"deleting a non-existing synthetic worked")

        self.assertFalse(category.DeleteTypeFilter(lldb.SBTypeNameSpecifier("")),"deleting a filter for '' worked")
        self.assertFalse(category.DeleteTypeSummary(lldb.SBTypeNameSpecifier("")),"deleting a summary for '' worked")
        self.assertFalse(category.DeleteTypeFormat(lldb.SBTypeNameSpecifier("")),"deleting a format for '' worked")
        self.assertFalse(category.DeleteTypeSynthetic(lldb.SBTypeNameSpecifier("")),"deleting a synthetic for '' worked")

        try:
             self.assertFalse(category.AddTypeSummary(lldb.SBTypeNameSpecifier("NoneSuchType"), None), "adding a summary valued None worked")
        except:
             pass
        else:
             self.assertFalse(True, "adding a summary valued None worked")

        try:
             self.assertFalse(category.AddTypeFilter(lldb.SBTypeNameSpecifier("NoneSuchType"), None), "adding a filter valued None worked")
        except:
             pass
        else:
             self.assertFalse(True, "adding a filter valued None worked")

        try:
             self.assertFalse(category.AddTypeSynthetic(lldb.SBTypeNameSpecifier("NoneSuchType"), None), "adding a synthetic valued None worked")
        except:
             pass
        else:
             self.assertFalse(True, "adding a synthetic valued None worked")

        try:
             self.assertFalse(category.AddTypeFormat(lldb.SBTypeNameSpecifier("NoneSuchType"), None), "adding a format valued None worked")
        except:
             pass
        else:
             self.assertFalse(True, "adding a format valued None worked")


        self.assertFalse(category.AddTypeSummary(lldb.SBTypeNameSpecifier("EmptySuchType"), lldb.SBTypeSummary()), "adding a summary without value worked")
        self.assertFalse(category.AddTypeFilter(lldb.SBTypeNameSpecifier("EmptySuchType"), lldb.SBTypeFilter()), "adding a filter without value worked")
        self.assertFalse(category.AddTypeSynthetic(lldb.SBTypeNameSpecifier("EmptySuchType"), lldb.SBTypeSynthetic()), "adding a synthetic without value worked")
        self.assertFalse(category.AddTypeFormat(lldb.SBTypeNameSpecifier("EmptySuchType"), lldb.SBTypeFormat()), "adding a format without value worked")

        self.assertFalse(category.AddTypeSummary(lldb.SBTypeNameSpecifier(""), lldb.SBTypeSummary.CreateWithSummaryString("")), "adding a summary for an invalid type worked")
        self.assertFalse(category.AddTypeFilter(lldb.SBTypeNameSpecifier(""), lldb.SBTypeFilter(0)), "adding a filter for an invalid type worked")
        self.assertFalse(category.AddTypeSynthetic(lldb.SBTypeNameSpecifier(""), lldb.SBTypeSynthetic.CreateWithClassName("")), "adding a synthetic for an invalid type worked")
        self.assertFalse(category.AddTypeFormat(lldb.SBTypeNameSpecifier(""), lldb.SBTypeFormat(lldb.eFormatHex)), "adding a format for an invalid type worked")

        new_category = self.dbg.CreateCategory("newbar")
        new_category.AddTypeSummary(lldb.SBTypeNameSpecifier("JustAStruct"),
             lldb.SBTypeSummary.CreateWithScriptCode("return 'hello scripted world';"))
        self.expect("frame variable foo", matching=False,
             substrs = ['hello scripted world'])
        new_category.enabled = True
        self.expect("frame variable foo", matching=True,
             substrs = ['hello scripted world'])

        self.expect("frame variable foo_ptr", matching=True,
             substrs = ['hello scripted world'])
        new_category.AddTypeSummary(lldb.SBTypeNameSpecifier("JustAStruct"),
             lldb.SBTypeSummary.CreateWithScriptCode("return 'hello scripted world';",
             lldb.eTypeOptionSkipPointers))
        self.expect("frame variable foo", matching=True,
             substrs = ['hello scripted world'])

        frame = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame()
        foo_ptr = frame.FindVariable("foo_ptr")
        summary = foo_ptr.GetTypeSummary()

        self.assertFalse(summary.IsValid(), "summary found for foo* when none was planned")

        self.expect("frame variable foo_ptr", matching=False,
             substrs = ['hello scripted world'])

        new_category.AddTypeSummary(lldb.SBTypeNameSpecifier("JustAStruct"),
             lldb.SBTypeSummary.CreateWithSummaryString("hello static world",
             lldb.eTypeOptionNone))

        summary = foo_ptr.GetTypeSummary()

        self.assertTrue(summary.IsValid(), "no summary found for foo* when one was in place")
        self.assertTrue(summary.GetData() == "hello static world", "wrong summary found for foo*")

        self.expect("frame variable e1", substrs=["I am an empty Empty1 {}"])
        self.expect("frame variable e2", substrs=["I am an empty Empty2"])
        self.expect("frame variable e2", substrs=["I am an empty Empty2 {}"], matching=False)
        
        self.assertTrue(self.dbg.GetCategory(lldb.eLanguageTypeObjC) is not None, "ObjC category is None")

    @add_test_categories(['pyapi'])
    def test_force_synth_off(self):
        """Test that one can have the public API return non-synthetic SBValues if desired"""
        self.build(dictionary={'EXE':'no_synth'})
        self.setTearDownCleanup()

        self.runCmd("file no_synth", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synthetic clear', check=False)
            self.runCmd('type category delete foobar', check=False)
            self.runCmd('type category delete JASSynth', check=False)
            self.runCmd('type category delete newbar', check=False)
            self.runCmd('settings set target.enable-synthetic-value true')

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        frame = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame()
        int_vector = frame.FindVariable("int_vector")
        if self.TraceOn():
             print(int_vector)
        self.assertTrue(int_vector.GetNumChildren() == 0, 'synthetic vector is empty')

        self.runCmd('settings set target.enable-synthetic-value false')
        frame = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame()
        int_vector = frame.FindVariable("int_vector")
        if self.TraceOn():
             print(int_vector)
        self.assertFalse(int_vector.GetNumChildren() == 0, '"physical" vector is not empty')

        self.runCmd('settings set target.enable-synthetic-value true')
        frame = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame()
        int_vector = frame.FindVariable("int_vector")
        if self.TraceOn():
             print(int_vector)
        self.assertTrue(int_vector.GetNumChildren() == 0, 'synthetic vector is still empty')
