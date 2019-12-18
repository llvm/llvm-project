"""
Test lldb-vscode setBreakpoints request
"""


import unittest2
import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import os


class TestVSCode_setBreakpoints(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows
    def test_set_and_clear(self):
        '''Tests setting and clearing source file and line breakpoints.
           This packet is a bit tricky on the debug adaptor side since there
           is no "clearBreakpoints" packet. Source file and line breakpoints
           are set by sending a "setBreakpoints" packet with a source file
           specified and zero or more source lines. If breakpoints have been
           set in the source file before, any exising breakpoints must remain
           set, and any new breakpoints must be created, and any breakpoints
           that were in previous requests and are not in the current request
           must be removed. This function tests this setting and clearing
           and makes sure things happen correctly. It doesn't test hitting
           breakpoints and the functionality of each breakpoint, like
           'conditions' and 'hitCondition' settings.'''
        source_basename = 'main.cpp'
        source_path = os.path.join(os.getcwd(), source_basename)
        first_line = line_number('main.cpp', 'break 12')
        second_line = line_number('main.cpp', 'break 13')
        third_line = line_number('main.cpp', 'break 14')
        lines = [first_line, second_line, third_line]

        # Visual Studio Code Debug Adaptors have no way to specify the file
        # without launching or attaching to a process, so we must start a
        # process in order to be able to set breakpoints.
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        # Set 3 breakoints and verify that they got set correctly
        response = self.vscode.request_setBreakpoints(source_path, lines)
        line_to_id = {}
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(lines),
                            "expect %u source breakpoints" % (len(lines)))
            for breakpoint in breakpoints:
                line = breakpoint['line']
                # Store the "id" of the breakpoint that was set for later
                line_to_id[line] = breakpoint['id']
                self.assertTrue(line in lines, "line expected in lines array")
                self.assertTrue(breakpoint['verified'],
                                "expect breakpoint verified")

        # There is no breakpoint delete packet, clients just send another
        # setBreakpoints packet with the same source file with fewer lines.
        # Below we remove the second line entry and call the setBreakpoints
        # function again. We want to verify that any breakpoints that were set
        # before still have the same "id". This means we didn't clear the
        # breakpoint and set it again at the same location. We also need to
        # verify that the second line location was actually removed.
        lines.remove(second_line)
        # Set 2 breakoints and verify that the previous breakoints that were
        # set above are still set.
        response = self.vscode.request_setBreakpoints(source_path, lines)
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(lines),
                            "expect %u source breakpoints" % (len(lines)))
            for breakpoint in breakpoints:
                line = breakpoint['line']
                # Verify the same breakpoints are still set within LLDB by
                # making sure the breakpoint ID didn't change
                self.assertTrue(line_to_id[line] == breakpoint['id'],
                                "verify previous breakpoints stayed the same")
                self.assertTrue(line in lines, "line expected in lines array")
                self.assertTrue(breakpoint['verified'],
                                "expect breakpoint still verified")

        # Now get the full list of breakpoints set in the target and verify
        # we have only 2 breakpoints set. The response above could have told
        # us about 2 breakpoints, but we want to make sure we don't have the
        # third one still set in the target
        response = self.vscode.request_testGetTargetBreakpoints()
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(lines),
                            "expect %u source breakpoints" % (len(lines)))
            for breakpoint in breakpoints:
                line = breakpoint['line']
                # Verify the same breakpoints are still set within LLDB by
                # making sure the breakpoint ID didn't change
                self.assertTrue(line_to_id[line] == breakpoint['id'],
                                "verify previous breakpoints stayed the same")
                self.assertTrue(line in lines, "line expected in lines array")
                self.assertTrue(breakpoint['verified'],
                                "expect breakpoint still verified")

        # Now clear all breakpoints for the source file by passing down an
        # empty lines array
        lines = []
        response = self.vscode.request_setBreakpoints(source_path, lines)
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(lines),
                            "expect %u source breakpoints" % (len(lines)))

        # Verify with the target that all breakpoints have been cleared
        response = self.vscode.request_testGetTargetBreakpoints()
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(lines),
                            "expect %u source breakpoints" % (len(lines)))

        # Now set a breakpoint again in the same source file and verify it
        # was added.
        lines = [second_line]
        response = self.vscode.request_setBreakpoints(source_path, lines)
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(lines),
                            "expect %u source breakpoints" % (len(lines)))
            for breakpoint in breakpoints:
                line = breakpoint['line']
                self.assertTrue(line in lines, "line expected in lines array")
                self.assertTrue(breakpoint['verified'],
                                "expect breakpoint still verified")

        # Now get the full list of breakpoints set in the target and verify
        # we have only 2 breakpoints set. The response above could have told
        # us about 2 breakpoints, but we want to make sure we don't have the
        # third one still set in the target
        response = self.vscode.request_testGetTargetBreakpoints()
        if response:
            breakpoints = response['body']['breakpoints']
            self.assertTrue(len(breakpoints) == len(lines),
                            "expect %u source breakpoints" % (len(lines)))
            for breakpoint in breakpoints:
                line = breakpoint['line']
                self.assertTrue(line in lines, "line expected in lines array")
                self.assertTrue(breakpoint['verified'],
                                "expect breakpoint still verified")

    @skipIfWindows
    def test_functionality(self):
        '''Tests hitting breakpoints and the functionality of a single
           breakpoint, like 'conditions' and 'hitCondition' settings.'''
        source_basename = 'main.cpp'
        source_path = os.path.join(os.getcwd(), source_basename)
        loop_line = line_number('main.cpp', '// break loop')

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        # Set a breakpoint at the loop line with no condition and no
        # hitCondition
        breakpoint_ids = self.set_source_breakpoints(source_path, [loop_line])
        self.assertTrue(len(breakpoint_ids) == 1, "expect one breakpoint")
        self.vscode.request_continue()

        # Verify we hit the breakpoint we just set
        self.verify_breakpoint_hit(breakpoint_ids)

        # Make sure i is zero at first breakpoint
        i = int(self.vscode.get_local_variable_value('i'))
        self.assertTrue(i == 0, 'i != 0 after hitting breakpoint')

        # Update the condition on our breakpoint
        new_breakpoint_ids = self.set_source_breakpoints(source_path,
                                                         [loop_line],
                                                         condition="i==4")
        self.assertTrue(breakpoint_ids == new_breakpoint_ids,
                        "existing breakpoint should have its condition "
                        "updated")

        self.continue_to_breakpoints(breakpoint_ids)
        i = int(self.vscode.get_local_variable_value('i'))
        self.assertTrue(i == 4,
                        'i != 4 showing conditional works')

        new_breakpoint_ids = self.set_source_breakpoints(source_path,
                                                         [loop_line],
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
