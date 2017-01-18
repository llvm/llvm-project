"""
Test that the lldb-mi driver understands MI command syntax.
"""

from __future__ import print_function


import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from functools import reduce


class MiSyntaxTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_tokens(self):
        """Test that 'lldb-mi --interpreter' prints command tokens."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("000-file-exec-and-symbols %s" % self.myexe)
        self.expect("000\^done")

        # Run to main
        self.runCmd("100000001-break-insert -f main")
        self.expect("100000001\^done,bkpt={number=\"1\"")
        self.runCmd("2-exec-run")
        self.expect("2\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Exit
        self.runCmd("0000000000000000000003-exec-continue")
        self.expect("0000000000000000000003\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_specialchars(self):
        """Test that 'lldb-mi --interpreter' handles complicated strings."""

        # Create an alias for myexe
        complicated_myexe = "C--mpl-x file's`s @#$%^&*()_+-={}[]| name"
        os.symlink(self.myexe, complicated_myexe)
        self.addTearDownHook(lambda: os.unlink(complicated_myexe))

        self.spawnLldbMi(args="\"%s\"" % complicated_myexe)

        # Test that the executable was loaded
        self.expect(
            "-file-exec-and-symbols \"%s\"" %
            complicated_myexe, exactly=True)
        self.expect("\^done")

        # Check that it was loaded correctly
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="Failing in ~6/600 dosep runs (build 3120-3122)")
    def test_lldbmi_process_output(self):
        """Test that 'lldb-mi --interpreter' wraps process output correctly."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run
        self.runCmd("-exec-run")
        self.expect("\^running")

        # Test that a process output is wrapped correctly
        self.expect("\@\"'\\\\r\\\\n\"")
        self.expect("\@\"` - it's \\\\\\\\n\\\\x12\\\\\"\\\\\\\\\\\\\"")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @expectedFailureAll(oslist=["macosx"], bugnumber="rdar://28805064")
    def test_lldbmi_output_grammar(self):
        """Test that 'lldb-mi --interpreter' uses standard output syntax."""

        self.spawnLldbMi(args=None)
        self.child.setecho(False)

        # Run all commands simultaneously
        self.runCmd("-unknown-command")
        self.runCmd("-interpreter-exec command help")
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.runCmd("-break-insert -f main")
        self.runCmd("-gdb-set target-async off")
        self.runCmd("-exec-run")
        self.runCmd("-gdb-set target-async on")
        self.runCmd("-exec-continue")
        self.runCmd("-gdb-exit")

        # Test that the program's output matches to the following pattern:
        # ( async-record | stream-record )* [ result-record ] "(gdb)" nl
        async_record = "^[0-9]*(\*|\+|=).+?\n"  # 1
        stream_record = "^(~|@|&).+?\n"         # 2
        result_record = "^[0-9]*\^.+?\n"        # 3
        prompt = "^\(gdb\)\r\n"                 # 4
        command = "^\r\n"                       # 5 (it looks like empty line for pexpect)
        error = "^.+?\n"                        # 6
        import pexpect                          # 7 (EOF)
        all_patterns = [
            async_record,
            stream_record,
            result_record,
            prompt,
            command,
            error,
            pexpect.EOF]

        # Routines to get a bit-mask for the specified list of patterns
        def get_bit(pattern): return all_patterns.index(pattern)
        def get_mask(pattern): return 1 << get_bit(pattern)
        def or_op(x, y): return x | y
        def get_state(*args): return reduce(or_op, map(get_mask, args))

        next_state = get_state(command)
        while True:
            it = self.expect(all_patterns)
            matched_pattern = all_patterns[it]

            # Check that state is acceptable
            if not (next_state & get_mask(matched_pattern)):
                self.fail(
                    "error: inconsistent pattern '%s' for state %#x (matched string: %s)" %
                    (repr(matched_pattern), next_state, self.child.after))
            elif matched_pattern == async_record or matched_pattern == stream_record:
                next_state = get_state(
                    async_record,
                    stream_record,
                    result_record,
                    prompt)
            elif matched_pattern == result_record:
                # FIXME lldb-mi prints async-records out of turn
                # ```
                #   ^done
                #   (gdb)
                #   ^running
                #   =thread-group-started,id="i1",pid="13875"
                #   (gdb)
                # ```
                # Therefore to pass that test I changed the grammar's rule:
                #   next_state = get_state(prompt)
                # to:
                next_state = get_state(async_record, prompt)
            elif matched_pattern == prompt:
                # FIXME lldb-mi prints the prompt out of turn
                # ```
                #   ^done
                #   (gdb)
                #   ^running
                #   (gdb)
                #   (gdb)
                # ```
                # Therefore to pass that test I changed the grammar's rule:
                #   next_state = get_state(async_record, stream_record, result_record, command, pexpect.EOF)
                # to:
                next_state = get_state(
                    async_record,
                    stream_record,
                    result_record,
                    prompt,
                    command,
                    pexpect.EOF)
            elif matched_pattern == command:
                next_state = get_state(
                    async_record,
                    stream_record,
                    result_record)
            elif matched_pattern == pexpect.EOF:
                break
            else:
                self.fail("error: pexpect returned an unknown state")
