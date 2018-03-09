"""
Test some target commands: create, list, select, variable.
"""

from __future__ import print_function


import lldb
import sys
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class targetCommandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers for our breakpoints.
        self.line_b = line_number('b.c', '// Set break point at this line.')
        self.line_c = line_number('c.c', '// Set break point at this line.')

    def test_target_command(self):
        """Test some target commands: create, list, select."""
        da = {'C_SOURCES': 'a.c', 'EXE': self.getBuildArtifact('a.out')}
        self.build(dictionary=da)
        self.addTearDownCleanup(dictionary=da)

        db = {'C_SOURCES': 'b.c', 'EXE': self.getBuildArtifact('b.out')}
        self.build(dictionary=db)
        self.addTearDownCleanup(dictionary=db)

        dc = {'C_SOURCES': 'c.c', 'EXE': self.getBuildArtifact('c.out')}
        self.build(dictionary=dc)
        self.addTearDownCleanup(dictionary=dc)

        self.do_target_command()

    # rdar://problem/9763907
    # 'target variable' command fails if the target program has been run
    @expectedFailureAndroid(archs=['aarch64'])
    def test_target_variable_command(self):
        """Test 'target variable' command before and after starting the inferior."""
        d = {'C_SOURCES': 'globals.c', 'EXE': self.getBuildArtifact('globals')}
        self.build(dictionary=d)
        self.addTearDownCleanup(dictionary=d)

        self.do_target_variable_command('globals')

    @expectedFailureAndroid(archs=['aarch64'])
    def test_target_variable_command_no_fail(self):
        """Test 'target variable' command before and after starting the inferior."""
        d = {'C_SOURCES': 'globals.c', 'EXE': self.getBuildArtifact('globals')}
        self.build(dictionary=d)
        self.addTearDownCleanup(dictionary=d)

        self.do_target_variable_command_no_fail('globals')

    def do_target_command(self):
        """Exercise 'target create', 'target list', 'target select' commands."""
        exe_a = self.getBuildArtifact("a.out")
        exe_b = self.getBuildArtifact("b.out")
        exe_c = self.getBuildArtifact("c.out")

        self.runCmd("target list")
        output = self.res.GetOutput()
        if output.startswith("No targets"):
            # We start from index 0.
            base = 0
        else:
            # Find the largest index of the existing list.
            import re
            pattern = re.compile("target #(\d+):")
            for line in reversed(output.split(os.linesep)):
                match = pattern.search(line)
                if match:
                    # We will start from (index + 1) ....
                    base = int(match.group(1), 10) + 1
                    #print("base is:", base)
                    break

        self.runCmd("target create " + exe_a, CURRENT_EXECUTABLE_SET)
        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("target create " + exe_b, CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, 'b.c', self.line_b, num_expected_locations=1, loc_exact=True)
        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("target create " + exe_c, CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, 'c.c', self.line_c, num_expected_locations=1, loc_exact=True)
        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("target list")

        self.runCmd("target select %d" % base)
        self.runCmd("thread backtrace")

        self.runCmd("target select %d" % (base + 2))
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['c.c:%d' % self.line_c,
                             'stop reason = breakpoint'])

        self.runCmd("target select %d" % (base + 1))
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['b.c:%d' % self.line_b,
                             'stop reason = breakpoint'])

        self.runCmd("target list")

    def do_target_variable_command(self, exe_name):
        """Exercise 'target variable' command before and after starting the inferior."""
        self.runCmd("file " + self.getBuildArtifact(exe_name),
                    CURRENT_EXECUTABLE_SET)

        self.expect(
            "target variable my_global_char",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "my_global_char",
                "'X'"])
        self.expect(
            "target variable my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_global_str',
                '"abc"'])
        self.expect(
            "target variable my_static_int",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_static_int',
                '228'])
        self.expect("target variable my_global_str_ptr", matching=False,
                    substrs=['"abc"'])
        self.expect("target variable *my_global_str_ptr", matching=True,
                    substrs=['"abc"'])
        self.expect(
            "target variable *my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=['a'])

        self.runCmd("b main")
        self.runCmd("run")

        self.expect(
            "target variable my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_global_str',
                '"abc"'])
        self.expect(
            "target variable my_static_int",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_static_int',
                '228'])
        self.expect("target variable my_global_str_ptr", matching=False,
                    substrs=['"abc"'])
        self.expect("target variable *my_global_str_ptr", matching=True,
                    substrs=['"abc"'])
        self.expect(
            "target variable *my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=['a'])
        self.expect(
            "target variable my_global_char",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "my_global_char",
                "'X'"])

        self.runCmd("c")

        # rdar://problem/9763907
        # 'target variable' command fails if the target program has been run
        self.expect(
            "target variable my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_global_str',
                '"abc"'])
        self.expect(
            "target variable my_static_int",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_static_int',
                '228'])
        self.expect("target variable my_global_str_ptr", matching=False,
                    substrs=['"abc"'])
        self.expect("target variable *my_global_str_ptr", matching=True,
                    substrs=['"abc"'])
        self.expect(
            "target variable *my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=['a'])
        self.expect(
            "target variable my_global_char",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "my_global_char",
                "'X'"])

    def do_target_variable_command_no_fail(self, exe_name):
        """Exercise 'target variable' command before and after starting the inferior."""
        self.runCmd("file " + self.getBuildArtifact(exe_name),
                    CURRENT_EXECUTABLE_SET)

        self.expect(
            "target variable my_global_char",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "my_global_char",
                "'X'"])
        self.expect(
            "target variable my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_global_str',
                '"abc"'])
        self.expect(
            "target variable my_static_int",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_static_int',
                '228'])
        self.expect("target variable my_global_str_ptr", matching=False,
                    substrs=['"abc"'])
        self.expect("target variable *my_global_str_ptr", matching=True,
                    substrs=['"abc"'])
        self.expect(
            "target variable *my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=['a'])

        self.runCmd("b main")
        self.runCmd("run")

        # New feature: you don't need to specify the variable(s) to 'target vaiable'.
        # It will find all the global and static variables in the current
        # compile unit.
        self.expect("target variable",
                    substrs=['my_global_char',
                             'my_global_str',
                             'my_global_str_ptr',
                             'my_static_int'])

        self.expect(
            "target variable my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_global_str',
                '"abc"'])
        self.expect(
            "target variable my_static_int",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'my_static_int',
                '228'])
        self.expect("target variable my_global_str_ptr", matching=False,
                    substrs=['"abc"'])
        self.expect("target variable *my_global_str_ptr", matching=True,
                    substrs=['"abc"'])
        self.expect(
            "target variable *my_global_str",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=['a'])
        self.expect(
            "target variable my_global_char",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "my_global_char",
                "'X'"])
