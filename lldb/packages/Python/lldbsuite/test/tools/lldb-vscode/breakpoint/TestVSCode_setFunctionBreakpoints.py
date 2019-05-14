"""
Test lldb-vscode setBreakpoints request
"""

from __future__ import print_function

import pprint
import unittest2
import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import os


class TestVSCode_setFunctionBreakpoints(
        lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows
    @skipIfDarwin # Skip this test for now until we can figure out why tings aren't working on build bots
    @no_debug_info_test
    def test_set_and_clear(self):
        '''Tests setting and clearing function breakpoints.
           This packet is a bit tricky on the debug adaptor side since there
           is no "clearFunction Breakpoints" packet. Function breakpoints
           are set by sending a "setFunctionBreakpoints" packet with zero or
           more function names. If function breakpoints have been set before,
           any exising breakpoints must remain set, and any new breakpoints
           must be created, and any breakpoints that were in previous requests
           and are not in the current request must be removed. This function
           tests this setting and clearing and makes sure things happen
           correctly. It doesn't test hitting breakpoints and the functionality
           of each breakpoint, like 'conditions' and 'hitCondition' settings.
        '''
        # Visual Studio Code Debug Adaptors have no way to specify the file
        # without launching or attaching to a process, so we must start a
        # process in order to be able to set breakpoints.
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        bp_id_12 = None
        functions = ['twelve']
        # Set a function breakpoint at 'twelve'
        response = self.vscode.request_setFunctionBreakpoints(functions)
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(functions),
                            "expect %u source breakpoints" % (len(functions)))
            for breakpoint in breakpoints:
                bp_id_12 = breakpoint['id']
                self.assertTrue(breakpoint['verified'],
                                "expect breakpoint verified")

        # Add an extra name and make sure we have two breakpoints after this
        functions.append('thirteen')
        response = self.vscode.request_setFunctionBreakpoints(functions)
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(functions),
                            "expect %u source breakpoints" % (len(functions)))
            for breakpoint in breakpoints:
                self.assertTrue(breakpoint['verified'],
                                "expect breakpoint verified")

        # There is no breakpoint delete packet, clients just send another
        # setFunctionBreakpoints packet with the different function names.
        functions.remove('thirteen')
        response = self.vscode.request_setFunctionBreakpoints(functions)
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(functions),
                            "expect %u source breakpoints" % (len(functions)))
            for breakpoint in breakpoints:
                bp_id = breakpoint['id']
                self.assertTrue(bp_id == bp_id_12,
                                'verify "twelve" breakpoint ID is same')
                self.assertTrue(breakpoint['verified'],
                                "expect breakpoint still verified")

        # Now get the full list of breakpoints set in the target and verify
        # we have only 1 breakpoints set. The response above could have told
        # us about 1 breakpoints, but we want to make sure we don't have the
        # second one still set in the target
        response = self.vscode.request_testGetTargetBreakpoints()
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(functions),
                            "expect %u source breakpoints" % (len(functions)))
            for breakpoint in breakpoints:
                bp_id = breakpoint['id']
                self.assertTrue(bp_id == bp_id_12,
                                'verify "twelve" breakpoint ID is same')
                self.assertTrue(breakpoint['verified'],
                                "expect breakpoint still verified")

        # Now clear all breakpoints for the source file by passing down an
        # empty lines array
        functions = []
        response = self.vscode.request_setFunctionBreakpoints(functions)
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(functions),
                            "expect %u source breakpoints" % (len(functions)))

        # Verify with the target that all breakpoints have been cleared
        response = self.vscode.request_testGetTargetBreakpoints()
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(functions),
                            "expect %u source breakpoints" % (len(functions)))

    @skipIfWindows
    @skipIfDarwin # Skip this test for now until we can figure out why tings aren't working on build bots
    @no_debug_info_test
    def test_functionality(self):
        '''Tests hitting breakpoints and the functionality of a single
           breakpoint, like 'conditions' and 'hitCondition' settings.'''

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        # Set a breakpoint on "twelve" with no condition and no hitCondition
        functions = ['twelve']
        breakpoint_ids = self.set_function_breakpoints(functions)

        self.assertTrue(len(breakpoint_ids) == len(functions),
                        "expect one breakpoint")

        # Verify we hit the breakpoint we just set
        self.continue_to_breakpoints(breakpoint_ids)

        # Make sure i is zero at first breakpoint
        i = int(self.vscode.get_local_variable_value('i'))
        self.assertTrue(i == 0, 'i != 0 after hitting breakpoint')

        # Update the condition on our breakpoint
        new_breakpoint_ids = self.set_function_breakpoints(functions,
                                                           condition="i==4")
        self.assertTrue(breakpoint_ids == new_breakpoint_ids,
                        "existing breakpoint should have its condition "
                        "updated")

        self.continue_to_breakpoints(breakpoint_ids)
        i = int(self.vscode.get_local_variable_value('i'))
        self.assertTrue(i == 4,
                        'i != 4 showing conditional works')
        new_breakpoint_ids = self.set_function_breakpoints(functions,
                                                           hitCondition="2")

        self.assertTrue(breakpoint_ids == new_breakpoint_ids,
                        "existing breakpoint should have its condition "
                        "updated")

        # Continue with a hitContidtion of 2 and expect it to skip 1 value
        self.continue_to_breakpoints(breakpoint_ids)
        i = int(self.vscode.get_local_variable_value('i'))
        self.assertTrue(i == 6,
                        'i != 6 showing hitCondition works')

        # continue after hitting our hitCondition and make sure it only goes
        # up by 1
        self.continue_to_breakpoints(breakpoint_ids)
        i = int(self.vscode.get_local_variable_value('i'))
        self.assertTrue(i == 7,
                        'i != 7 showing post hitCondition hits every time')
