"""Test the lldb public C++ api breakpoint callbacks."""

from __future__ import print_function

# __package__ = "lldbsuite.test"


import os
import re
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import subprocess
import unittest2


@unittest2.skip("skipping due to frequent timeouts: rdar://28183131")
class SBBreakpointCallbackCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.generateSource('driver.cpp')
        self.generateSource('listener_test.cpp')
        self.generateSource('test_breakpoint_callback.cpp')
        self.generateSource('test_listener_event_description.cpp')
        self.generateSource('test_listener_event_process_state.cpp')
        self.generateSource('test_listener_resume.cpp')

    mydir = TestBase.compute_mydir(__file__)

    @skipIfRemote
    @skipIfNoSBHeaders
    # clang-cl does not support throw or catch (llvm.org/pr24538)
    @skipIfWindows
    def test_breakpoint_callback(self):
        """Test the that SBBreakpoint callback is invoked when a breakpoint is hit. """
        self.build_and_test('driver.cpp test_breakpoint_callback.cpp',
                            'test_breakpoint_callback')

    @skipIfRemote
    @skipIfNoSBHeaders
    # clang-cl does not support throw or catch (llvm.org/pr24538)
    @skipIfWindows
    @expectedFlakeyFreeBSD
    def test_sb_api_listener_event_description(self):
        """ Test the description of an SBListener breakpoint event is valid."""
        self.build_and_test(
            'driver.cpp listener_test.cpp test_listener_event_description.cpp',
            'test_listener_event_description')
        pass

    @skipIfRemote
    @skipIfNoSBHeaders
    # clang-cl does not support throw or catch (llvm.org/pr24538)
    @skipIfWindows
    @expectedFlakeyFreeBSD
    def test_sb_api_listener_event_process_state(self):
        """ Test that a registered SBListener receives events when a process
            changes state.
        """
        self.build_and_test(
            'driver.cpp listener_test.cpp test_listener_event_process_state.cpp',
            'test_listener_event_process_state')
        pass

    @skipIfRemote
    @skipIfNoSBHeaders
    # clang-cl does not support throw or catch (llvm.org/pr24538)
    @skipIfWindows
    @expectedFlakeyFreeBSD
    @skipIf(oslist=["linux"]) # flakey
    def test_sb_api_listener_resume(self):
        """ Test that a process can be resumed from a non-main thread. """
        self.build_and_test(
            'driver.cpp listener_test.cpp test_listener_resume.cpp',
            'test_listener_resume')
        pass

    def build_and_test(self, sources, test_name, args=None):
        """ Build LLDB test from sources, and run expecting 0 exit code """

        # These tests link against host lldb API.
        # Compiler's target triple must match liblldb triple
        # because remote is disabled, we can assume that the os is the same
        # still need to check architecture
        if self.getLldbArchitecture() != self.getArchitecture():
            self.skipTest(
                "This test is only run if the target arch is the same as the lldb binary arch")

        self.inferior = 'inferior_program'
        self.buildProgram('inferior.cpp', self.inferior)
        self.addTearDownHook(lambda:
                             os.remove(self.getBuildArtifact(self.inferior)))

        self.buildDriver(sources, test_name)
        self.addTearDownHook(lambda:
                             os.remove(self.getBuildArtifact(test_name)))

        test_exe = self.getBuildArtifact(test_name)
        self.signBinary(test_exe)
        exe = [test_exe, self.getBuildArtifact(self.inferior)]

        env = {self.dylibPath: self.getLLDBLibraryEnvVal()}
        if 'LLDB_DEBUGSERVER_PATH' in os.environ:
            env['LLDB_DEBUGSERVER_PATH'] = os.environ['LLDB_DEBUGSERVER_PATH']
        if self.TraceOn():
            print("Running test %s" % " ".join(exe))
            check_call(exe, env=env)
        else:
            with open(os.devnull, 'w') as fnull:
                check_call(exe, env=env, stdout=fnull, stderr=fnull)

    def build_program(self, sources, program):
        return self.buildDriver(sources, program)
