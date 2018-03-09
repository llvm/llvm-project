"""
Test lldb target stop-hook mechanism to see whether it fires off correctly .
"""

from __future__ import print_function


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import configuration
from lldbsuite.test import lldbutil


class StopHookMechanismTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers inside main.cpp.
        self.begl = line_number(
            'main.cpp',
            '// Set breakpoint here to test target stop-hook.')
        self.endl = line_number(
            'main.cpp',
            '// End of the line range for which stop-hook is to be run.')
        self.correct_step_line = line_number(
            'main.cpp', '// We should stop here after stepping.')
        self.line = line_number(
            'main.cpp',
            '// Another breakpoint which is outside of the stop-hook range.')

    @skipIfFreeBSD  # llvm.org/pr15037
    # stop-hooks sometimes fail to fire on Linux
    @expectedFlakeyLinux('llvm.org/pr15037')
    @expectedFailureAll(
        hostoslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIf(oslist=['ios', 'watchos', 'tvos', 'bridgeos'], archs=['armv7', 'armv7k'])  # <rdar://problem/34582291> problem with armv7 and step-over and stop-hook firing on ios etc systems
    def test(self):
        """Test the stop-hook mechanism."""
        self.build()

        import pexpect
        exe = self.getBuildArtifact("a.out")
        prompt = "(lldb) "
        add_prompt = "Enter your stop hook command(s).  Type 'DONE' to end."
        add_prompt1 = "> "

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s %s' %
                                   (lldbtest_config.lldbExec, self.lldbOption))
        child = self.child
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        if lldb.remote_platform:
            child.expect_exact(prompt)
            child.sendline(
                'platform select %s' %
                lldb.remote_platform.GetName())
            child.expect_exact(prompt)
            child.sendline(
                'platform connect %s' %
                configuration.lldb_platform_url)
            child.expect_exact(prompt)
            child.sendline(
                'platform settings -w %s' %
                configuration.lldb_platform_working_dir)

        child.expect_exact(prompt)
        child.sendline('target create %s' % exe)

        # Set the breakpoint, followed by the target stop-hook commands.
        child.expect_exact(prompt)
        child.sendline('breakpoint set -f main.cpp -l %d' % self.begl)
        child.expect_exact(prompt)
        child.sendline('breakpoint set -f main.cpp -l %d' % self.line)
        child.expect_exact(prompt)
        child.sendline(
            'target stop-hook add -f main.cpp -l %d -e %d' %
            (self.begl, self.endl))
        child.expect_exact(add_prompt)
        child.expect_exact(add_prompt1)
        child.sendline('expr ptr')
        child.expect_exact(add_prompt1)
        child.sendline('DONE')
        child.expect_exact(prompt)
        child.sendline('target stop-hook list')

        # Now run the program, expect to stop at the first breakpoint which is
        # within the stop-hook range.
        child.expect_exact(prompt)
        child.sendline('run')
        # Make sure we see the stop hook text from the stop of the process from
        # the run hitting the first breakpoint
        child.expect_exact('(void *) $')
        child.expect_exact(prompt)
        child.sendline('thread step-over')
        # Expecting to find the output emitted by the firing of our stop hook.
        child.expect_exact('(void *) $')
        # This is orthogonal to the main stop hook test, but this example shows a bug in
        # CLANG where the line table entry for the "return -1" actually includes some code
        # from the other branch of the if/else, so we incorrectly stop at the "return -1" line.
        # I fixed that in lldb and I'm sticking in a test here because I don't want to have to
        # make up a whole nother test case for it.
        child.sendline('frame info')
        at_line = 'at main.cpp:%d' % (self.correct_step_line)
        print('expecting "%s"' % at_line)
        child.expect_exact(at_line)

        # Now continue the inferior, we'll stop at another breakpoint which is
        # outside the stop-hook range.
        child.sendline('process continue')
        child.expect_exact(
            '// Another breakpoint which is outside of the stop-hook range.')
        # self.DebugPExpect(child)
        child.sendline('thread step-over')
        child.expect_exact(
            '// Another breakpoint which is outside of the stop-hook range.')
        # self.DebugPExpect(child)
        # Verify that the 'Stop Hooks' mechanism is NOT BEING fired off.
        self.expect(child.before, exe=False, matching=False,
                    substrs=['(void *) $'])
