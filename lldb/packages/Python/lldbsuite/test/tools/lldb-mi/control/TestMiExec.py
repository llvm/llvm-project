"""
Test lldb-mi -exec-xxx commands.
"""

from __future__ import print_function



import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class MiExecTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @expectedFailureLinux # llvm.org/pr25000: lldb-mi does not receive broadcasted notification from Core/Process about process stopped
    def test_lldbmi_exec_run(self):
        """Test that 'lldb-mi --interpreter' can stop at entry."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Test that program is stopped at entry
        self.runCmd("-exec-run --start")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"signal-received\",signal-name=\"SIGSTOP\",signal-meaning=\"Stop\",.*?thread-id=\"1\",stopped-threads=\"all\"")
        # Test that lldb-mi is ready to execute next commands
        self.expect(self.child_prompt, exactly = True)

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_abort(self):
        """Test that 'lldb-mi --interpreter' works for -exec-abort."""

        self.spawnLldbMi(args = None)

        # Test that -exec-abort fails on invalid process
        self.runCmd("-exec-abort")
        self.expect("\^error,msg=\"Command 'exec-abort'\. Invalid process during debug session\"")

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Set arguments
        self.runCmd("-exec-arguments arg1")
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that arguments were passed
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"2\"")

        # Test that program may be aborted
        self.runCmd("-exec-abort")
        self.expect("\^done")
        self.expect("\*stopped,reason=\"exited-normally\"")

        # Test that program can be run again
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that arguments were passed again
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"2\"")

        # Test that program may be aborted again
        self.runCmd("-exec-abort")
        self.expect("\^done")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_arguments_set(self):
        """Test that 'lldb-mi --interpreter' can pass args using -exec-arguments."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Set arguments
        self.runCmd("-exec-arguments --arg1 \"2nd arg\" third_arg fourth=\"4th arg\"")
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Check argc and argv to see if arg passed
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"5\"")
        #self.runCmd("-data-evaluate-expression argv[1]")
        #self.expect("\^done,value=\"--arg1\"")
        self.runCmd("-interpreter-exec command \"print argv[1]\"")
        self.expect("\"--arg1\"")
        #self.runCmd("-data-evaluate-expression argv[2]")
        #self.expect("\^done,value=\"2nd arg\"")
        self.runCmd("-interpreter-exec command \"print argv[2]\"")
        self.expect("\"2nd arg\"")
        #self.runCmd("-data-evaluate-expression argv[3]")
        #self.expect("\^done,value=\"third_arg\"")
        self.runCmd("-interpreter-exec command \"print argv[3]\"")
        self.expect("\"third_arg\"")
        #self.runCmd("-data-evaluate-expression argv[4]")
        #self.expect("\^done,value=\"fourth=\\\\\\\"4th arg\\\\\\\"\"")
        self.runCmd("-interpreter-exec command \"print argv[4]\"")
        self.expect("\"fourth=\\\\\\\"4th arg\\\\\\\"\"")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_arguments_reset(self):
        """Test that 'lldb-mi --interpreter' can reset previously set args using -exec-arguments."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Set arguments
        self.runCmd("-exec-arguments arg1")
        self.expect("\^done")
        self.runCmd("-exec-arguments")
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Check argc to see if arg passed
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"1\"")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_next(self):
        """Test that 'lldb-mi --interpreter' works for stepping."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Warning: the following is sensitive to the lines in the source

        # Test -exec-next
        self.runCmd("-exec-next --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?main\.cpp\",line=\"29\"")

        # Test that --thread is optional
        self.runCmd("-exec-next --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?main\.cpp\",line=\"30\"")

        # Test that --frame is optional
        self.runCmd("-exec-next --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?main\.cpp\",line=\"31\"")

        # Test that both --thread and --frame are optional
        self.runCmd("-exec-next")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?main\.cpp\",line=\"32\"")

        # Test that an invalid --thread is handled
        self.runCmd("-exec-next --thread 0")
        self.expect("\^error,message=\"error: Thread index 0 is out of range")
        self.runCmd("-exec-next --thread 10")
        self.expect("\^error,message=\"error: Thread index 10 is out of range")

        # Test that an invalid --frame is handled
        # FIXME: no error is returned
        self.runCmd("-exec-next --frame 10")
        #self.expect("\^error: Frame index 10 is out of range")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @expectedFailurei386 #xfail to get buildbot green, failing config: i386 binary running on ubuntu 14.04 x86_64
    def test_lldbmi_exec_next_instruction(self):
        """Test that 'lldb-mi --interpreter' works for instruction stepping."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Warning: the following is sensitive to the lines in the
        # source and optimizations

        # Test -exec-next-instruction
        self.runCmd("-exec-next-instruction --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?main\.cpp\",line=\"28\"")

        # Test that --thread is optional
        self.runCmd("-exec-next-instruction --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?main\.cpp\",line=\"28\"")

        # Test that --frame is optional
        self.runCmd("-exec-next-instruction --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?main\.cpp\",line=\"29\"")

        # Test that both --thread and --frame are optional
        self.runCmd("-exec-next-instruction")
        self.expect("\^running")
        # Depending on compiler, it can stop at different line
        self.expect("\*stopped,reason=\"end-stepping-range\".+?main\.cpp\",line=\"(29|30)\"")

        # Test that an invalid --thread is handled
        self.runCmd("-exec-next-instruction --thread 0")
        self.expect("\^error,message=\"error: Thread index 0 is out of range")
        self.runCmd("-exec-next-instruction --thread 10")
        self.expect("\^error,message=\"error: Thread index 10 is out of range")

        # Test that an invalid --frame is handled
        # FIXME: no error is returned
        self.runCmd("-exec-next-instruction --frame 10")
        #self.expect("\^error: Frame index 10 is out of range")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_step(self):
        """Test that 'lldb-mi --interpreter' works for stepping into."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Warning: the following is sensitive to the lines in the source

        # Test that -exec-step steps into (or not) printf depending on debug info
        # Note that message is different in Darwin and Linux:
        # Darwin: "*stopped,reason=\"end-stepping-range\",frame={addr=\"0x[0-9a-f]+\",func=\"main\",args=[{name=\"argc\",value=\"1\"},{name=\"argv\",value="0x[0-9a-f]+\"}],file=\"main.cpp\",fullname=\".+main.cpp\",line=\"\d\"},thread-id=\"1\",stopped-threads=\"all\"
        # Linux:  "*stopped,reason=\"end-stepping-range\",frame={addr="0x[0-9a-f]+\",func=\"__printf\",args=[{name=\"format\",value=\"0x[0-9a-f]+\"}],file=\"printf.c\",fullname=\".+printf.c\",line="\d+"},thread-id=\"1\",stopped-threads=\"all\"
        self.runCmd("-exec-step --thread 1 --frame 0")
        self.expect("\^running")
        it = self.expect([ "\*stopped,reason=\"end-stepping-range\".+?func=\"main\"",
                           "\*stopped,reason=\"end-stepping-range\".+?func=\"(?!main).+?\"" ])
        # Exit from printf if needed
        if it == 1:
            self.runCmd("-exec-finish")
            self.expect("\^running")
            self.expect("\*stopped,reason=\"end-stepping-range\".+?func=\"main\"")

        # Test that -exec-step steps into g_MyFunction and back out
        # (and that --thread is optional)
        self.runCmd("-exec-step --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?func=\"g_MyFunction.*?\"")
        # Use -exec-finish here to make sure that control reaches the caller.
        # -exec-step can keep us in the g_MyFunction for gcc
        self.runCmd("-exec-finish --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?main\.cpp\",line=\"30\"")

        # Test that -exec-step steps into s_MyFunction
        # (and that --frame is optional)
        self.runCmd("-exec-step --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?func=\".*?s_MyFunction.*?\"")

        # Test that -exec-step steps into g_MyFunction from inside
        # s_MyFunction (and that both --thread and --frame are optional)
        self.runCmd("-exec-step")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?func=\"g_MyFunction.*?\"")

        # Test that an invalid --thread is handled
        self.runCmd("-exec-step --thread 0")
        self.expect("\^error,message=\"error: Thread index 0 is out of range")
        self.runCmd("-exec-step --thread 10")
        self.expect("\^error,message=\"error: Thread index 10 is out of range")

        # Test that an invalid --frame is handled
        # FIXME: no error is returned
        self.runCmd("-exec-step --frame 10")
        #self.expect("\^error: Frame index 10 is out of range")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_step_instruction(self):
        """Test that 'lldb-mi --interpreter' works for instruction stepping into."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Warning: the following is sensitive to the lines in the
        # source and optimizations

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -exec-next steps over printf
        self.runCmd("-exec-next --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?main\.cpp\",line=\"29\"")

        # Test that -exec-step-instruction steps over non branching
        # instruction
        self.runCmd("-exec-step-instruction --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?main\.cpp\"")

        # Test that -exec-step-instruction steps into g_MyFunction
        # instruction (and that --thread is optional)
        self.runCmd("-exec-step-instruction --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?func=\"g_MyFunction.*?\"")

        # Test that -exec-step-instruction steps over non branching
        # (and that --frame is optional)
        self.runCmd("-exec-step-instruction --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?func=\"g_MyFunction.*?\"")

        # Test that -exec-step-instruction steps into g_MyFunction
        # (and that both --thread and --frame are optional)
        self.runCmd("-exec-step-instruction")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?func=\"g_MyFunction.*?\"")

        # Test that an invalid --thread is handled
        self.runCmd("-exec-step-instruction --thread 0")
        self.expect("\^error,message=\"error: Thread index 0 is out of range")
        self.runCmd("-exec-step-instruction --thread 10")
        self.expect("\^error,message=\"error: Thread index 10 is out of range")

        # Test that an invalid --frame is handled
        # FIXME: no error is returned
        self.runCmd("-exec-step-instruction --frame 10")
        #self.expect("\^error: Frame index 10 is out of range")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_finish(self):
        """Test that 'lldb-mi --interpreter' works for -exec-finish."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Set BP at g_MyFunction and run to BP
        self.runCmd("-break-insert -f g_MyFunction")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -exec-finish returns from g_MyFunction
        self.runCmd("-exec-finish --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?func=\"main\"")

        # Run to BP inside s_MyFunction call
        self.runCmd("-break-insert s_MyFunction")
        self.expect("\^done,bkpt={number=\"2\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -exec-finish hits BP at g_MyFunction call inside
        # s_MyFunction (and that --thread is optional)
        self.runCmd("-exec-finish --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -exec-finish returns from g_MyFunction call inside
        # s_MyFunction (and that --frame is optional)
        self.runCmd("-exec-finish --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?func=\".*?s_MyFunction.*?\"")

        # Test that -exec-finish returns from s_MyFunction
        # (and that both --thread and --frame are optional)
        self.runCmd("-exec-finish")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?func=\"main\"")

        # Test that an invalid --thread is handled
        self.runCmd("-exec-finish --thread 0")
        self.expect("\^error,message=\"error: Thread index 0 is out of range")
        self.runCmd("-exec-finish --thread 10")
        self.expect("\^error,message=\"error: Thread index 10 is out of range")

        # Test that an invalid --frame is handled
        # FIXME: no error is returned
        #self.runCmd("-exec-finish --frame 10")
        #self.expect("\^error: Frame index 10 is out of range")

        # Set BP at printf and run to BP
        self.runCmd("-break-insert -f printf")
        self.expect("\^done,bkpt={number=\"3\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        ## Test that -exec-finish returns from printf
        self.runCmd("-exec-finish --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".+?func=\"main\"")
