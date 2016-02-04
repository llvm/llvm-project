"""
Test displayed value of a vector variable while doing watchpoint operations
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestValueOfVectorVariableTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAndroid(archs=['arm', 'aarch64']) # Watchpoints not supported
    @expectedFailureWindows("llvm.org/pr24446") # WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows
    def test_value_of_vector_variable_using_watchpoint_set(self):
        """Test verify displayed value of vector variable."""
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.value_of_vector_variable_with_watchpoint_set()
    
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'
        self.exe_name = 'a.out'
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

    def value_of_vector_variable_with_watchpoint_set(self):
        """Test verify displayed value of vector variable"""
        exe = os.path.join(os.getcwd(), 'a.out')
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set break to get a frame
        self.runCmd("b main")
            
        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Value of a vector variable should be displayed correctly
        self.expect("watchpoint set variable global_vector", WATCHPOINT_CREATED,
            substrs = ['new value: (1, 2, 3, 4)'])
