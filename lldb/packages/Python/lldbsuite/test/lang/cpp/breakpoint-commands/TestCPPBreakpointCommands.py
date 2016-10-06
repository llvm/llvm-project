"""
Test lldb breakpoint command for CPP methods & functions in a namespace.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CPPBreakpointCommandsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def make_breakpoint(self, name, type, expected_num_locations):
        bkpt = self.target.BreakpointCreateByName(name,
                                                  type,
                                                  self.a_out_module,
                                                  self.nested_comp_unit)
        num_locations = bkpt.GetNumLocations()
        self.assertTrue(
            num_locations == expected_num_locations,
            "Wrong number of locations for '%s', expected: %d got: %d" %
            (name,
             expected_num_locations,
             num_locations))
        return bkpt

    def test_cpp_breakpoint_cmds(self):
        """Test a sequence of breakpoint command add, list, and delete."""
        self.build()

        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target from the debugger.

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        self.a_out_module = lldb.SBFileSpecList()
        self.a_out_module.Append(lldb.SBFileSpec(exe))

        self.nested_comp_unit = lldb.SBFileSpecList()
        self.nested_comp_unit.Append(lldb.SBFileSpec("nested.cpp"))

        # First provide ONLY the method name.  This should get everybody...
        self.make_breakpoint("Function",
                             lldb.eFunctionNameTypeAuto,
                             5)

        # Now add the Baz class specifier.  This should get the version contained in Bar,
        # AND the one contained in ::
        self.make_breakpoint("Baz::Function",
                             lldb.eFunctionNameTypeAuto,
                             2)

        # Then add the Bar::Baz specifier.  This should get the version
        # contained in Bar only
        self.make_breakpoint("Bar::Baz::Function",
                             lldb.eFunctionNameTypeAuto,
                             1)

        self.make_breakpoint("Function",
                             lldb.eFunctionNameTypeMethod,
                             3)

        self.make_breakpoint("Baz::Function",
                             lldb.eFunctionNameTypeMethod,
                             2)

        self.make_breakpoint("Bar::Baz::Function",
                             lldb.eFunctionNameTypeMethod,
                             1)

        self.make_breakpoint("Function",
                             lldb.eFunctionNameTypeBase,
                             2)

        self.make_breakpoint("Bar::Function",
                             lldb.eFunctionNameTypeBase,
                             1)
