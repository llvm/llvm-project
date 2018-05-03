"""
Test that we can lookup types correctly in the expression parser
"""

from __future__ import print_function


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test import decorators

class TestCppTypeLookup(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def check_value(self, value, ivar_name, ivar_value):
        self.assertTrue(value.GetError().Success(),
                        "Invalid valobj: %s" % (
                                value.GetError().GetCString()))
        ivar = value.GetChildMemberWithName(ivar_name)
        self.assertTrue(ivar.GetError().Success(),
                        "Failed to fetch ivar named '%s'" % (ivar_name))
        self.assertEqual(ivar_value,
                         ivar.GetValueAsSigned(),
                         "Got the right value for ivar")

    def test_namespace_only(self):
        """
            Test that we fail to lookup a struct type that exists only in a
            namespace.
        """
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file)

        # Get frame for current thread
        frame = thread.GetSelectedFrame()

        # Make sure we don't accidentally accept structures that exist only
        # in namespaces when evaluating expressions with top level types.
        # Prior to the revision that added this test, we would accidentally
        # accept types from namespaces, so this will ensure we don't regress
        # to that behavior again
        expr_result = frame.EvaluateExpression("*((namespace_only *)&i)")
        self.assertTrue(expr_result.GetError().Fail(),
                        "'namespace_only' exists in namespace only")

        # Make sure we can find the correct type in a namespace "a"
        expr_result = frame.EvaluateExpression("*((a::namespace_only *)&i)")
        self.check_value(expr_result, "a", 123)
        # Make sure we can find the correct type in a namespace "b"
        expr_result = frame.EvaluateExpression("*((b::namespace_only *)&i)")
        self.check_value(expr_result, "b", 123)

        # Make sure we can find the correct type in the root namespace
        expr_result = frame.EvaluateExpression("*((namespace_and_file *)&i)")
        self.check_value(expr_result, "ff", 123)
        # Make sure we can find the correct type in a namespace "a"
        expr_result = frame.EvaluateExpression(
                "*((a::namespace_and_file *)&i)")
        self.check_value(expr_result, "aa", 123)
        # Make sure we can find the correct type in a namespace "b"
        expr_result = frame.EvaluateExpression(
                "*((b::namespace_and_file *)&i)")
        self.check_value(expr_result, "bb", 123)

        # Make sure we don't accidentally accept structures that exist only
        # in namespaces when evaluating expressions with top level types.
        # Prior to the revision that added this test, we would accidentally
        # accept types from namespaces, so this will ensure we don't regress
        # to that behavior again
        expr_result = frame.EvaluateExpression("*((in_contains_type *)&i)")
        self.assertTrue(expr_result.GetError().Fail(),
                        "'in_contains_type' exists in struct only")

        # Make sure we can find the correct type in the root namespace
        expr_result = frame.EvaluateExpression(
                "*((contains_type::in_contains_type *)&i)")
        self.check_value(expr_result, "fff", 123)
        # Make sure we can find the correct type in a namespace "a"
        expr_result = frame.EvaluateExpression(
                "*((a::contains_type::in_contains_type *)&i)")
        self.check_value(expr_result, "aaa", 123)
        # Make sure we can find the correct type in a namespace "b"
        expr_result = frame.EvaluateExpression(
                "*((b::contains_type::in_contains_type *)&i)")
        self.check_value(expr_result, "bbb", 123)
