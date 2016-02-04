"""
Test that lldb stop-hook works for multiple threads.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import configuration
from lldbsuite.test import lldbutil

class StopHookForMultipleThreadsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.first_stop = line_number(self.source, '// Set break point at this line, and add a stop-hook.')
        self.thread_function = line_number(self.source, '// Break here to test that the stop-hook mechanism works for multiple threads.')
        # Build dictionary to have unique executable names for each test method.
        self.exe_name = self.testMethodName
        self.d = {'CXX_SOURCES': self.source, 'EXE': self.exe_name}

    @expectedFlakeyFreeBSD("llvm.org/pr15037")
    @expectedFlakeyLinux("llvm.org/pr15037") # stop hooks sometimes fail to fire on Linux
    @expectedFailureHostWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_stop_hook_multiple_threads(self):
        """Test that lldb stop-hook works for multiple threads."""
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        import pexpect
        exe = os.path.join(os.getcwd(), self.exe_name)
        prompt = "(lldb) "

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s %s' % (lldbtest_config.lldbExec, self.lldbOption))
        child = self.child
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        if lldb.remote_platform:
            child.expect_exact(prompt)
            child.sendline('platform select %s' % lldb.remote_platform.GetName())
            child.expect_exact(prompt)
            child.sendline('platform connect %s' % configuration.lldb_platform_url)
            child.expect_exact(prompt)
            child.sendline('platform settings -w %s' % configuration.lldb_platform_working_dir)

        child.expect_exact(prompt)
        child.sendline('target create %s' % exe)

        # Set the breakpoint, followed by the target stop-hook commands.
        child.expect_exact(prompt)
        child.sendline('breakpoint set -f main.cpp -l %d' % self.first_stop)
        child.expect_exact(prompt)
        child.sendline('breakpoint set -f main.cpp -l %d' % self.thread_function)
        child.expect_exact(prompt)

        # Now run the program, expect to stop at the first breakpoint which is within the stop-hook range.
        child.sendline('run')
        child.expect_exact("Process")   # 'Process 2415 launched', 'Process 2415 stopped'
        child.expect_exact(prompt)
        child.sendline('target stop-hook add -o "frame variable --show-globals g_val"')
        child.expect_exact("Stop hook") # 'Stop hook #1 added.'
        child.expect_exact(prompt)

        # Continue and expect to find the output emitted by the firing of our stop hook.
        child.sendline('continue')
        child.expect_exact('(uint32_t) ::g_val = ')
