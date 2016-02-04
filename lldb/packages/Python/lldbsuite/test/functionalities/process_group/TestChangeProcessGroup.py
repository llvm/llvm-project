"""Test that we handle inferiors which change their process group"""

from __future__ import print_function



import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ChangeProcessGroupTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.c', '// Set breakpoint here')

    @skipIfFreeBSD # Times out on FreeBSD llvm.org/pr23731
    @skipIfWindows # setpgid call does not exist on Windows
    @expectedFailureAndroid("http://llvm.org/pr23762", api_levels=[16])
    def test_setpgid(self):
        self.build()
        exe = os.path.join(os.getcwd(), 'a.out')

        # Use a file as a synchronization point between test and inferior.
        pid_file_path = lldbutil.append_to_process_working_directory(
                "pid_file_%d" % (int(time.time())))
        self.addTearDownHook(lambda: self.run_platform_command("rm %s" % (pid_file_path)))

        popen = self.spawnSubprocess(exe, [pid_file_path])
        self.addTearDownHook(self.cleanupSubprocesses)

        max_attempts = 5
        for i in range(max_attempts):
            err, retcode, msg = self.run_platform_command("ls %s" % pid_file_path)
            if err.Success() and retcode == 0:
                break
            else:
                print(msg)
            if i < max_attempts:
                # Exponential backoff!
                time.sleep(pow(2, i) * 0.25)
        else:
            self.fail("Child PID file %s not found even after %d attempts." % (pid_file_path, max_attempts))

        err, retcode, pid = self.run_platform_command("cat %s" % (pid_file_path))

        self.assertTrue(err.Success() and retcode == 0,
                "Failed to read file %s: %s, retcode: %d" % (pid_file_path, err.GetCString(), retcode))

        # make sure we cleanup the forked child also
        def cleanupChild():
            if lldb.remote_platform:
                lldb.remote_platform.Kill(int(pid))
            else:
                if os.path.exists("/proc/" + pid):
                    os.kill(int(pid), signal.SIGKILL)
        self.addTearDownHook(cleanupChild)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        listener = lldb.SBListener("my.attach.listener")
        error = lldb.SBError()
        process = target.AttachToProcessWithID(listener, int(pid), error)
        self.assertTrue(error.Success() and process, PROCESS_IS_VALID)

        # set a breakpoint just before the setpgid() call
        lldbutil.run_break_set_by_file_and_line(self, 'main.c', self.line, num_expected_locations=-1)

        thread = process.GetSelectedThread()

        # release the child from its loop
        value = thread.GetSelectedFrame().EvaluateExpression("release_child_flag = 1")
        self.assertTrue(value.IsValid() and value.GetValueAsUnsigned(0) == 1);
        process.Continue()

        # make sure the child's process group id is different from its pid
        value = thread.GetSelectedFrame().EvaluateExpression("(int)getpgid(0)")
        self.assertTrue(value.IsValid())
        self.assertNotEqual(value.GetValueAsUnsigned(0), int(pid));

        # step over the setpgid() call
        thread.StepOver()
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        # verify that the process group has been set correctly
        # this also checks that we are still in full control of the child
        value = thread.GetSelectedFrame().EvaluateExpression("(int)getpgid(0)")
        self.assertTrue(value.IsValid())
        self.assertEqual(value.GetValueAsUnsigned(0), int(pid));

        # run to completion
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)

    def run_platform_command(self, cmd):
        platform = self.dbg.GetSelectedPlatform()
        shell_command = lldb.SBPlatformShellCommand(cmd)
        err = platform.Run(shell_command)
        return (err, shell_command.GetStatus(), shell_command.GetOutput())
