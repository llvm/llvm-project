"""
Test the lldb command line completion mechanism.
"""

from __future__ import print_function


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbplatform
from lldbsuite.test import lldbutil


class CommandLineCompletionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproducts."""
        try:
            os.remove("child_send.txt")
            os.remove("child_read.txt")
        except:
            pass

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_at(self):
        """Test that 'at' completes to 'attach '."""
        self.complete_from_to('at', 'attach ')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_de(self):
        """Test that 'de' completes to 'detach '."""
        self.complete_from_to('de', 'detach ')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_frame_variable(self):
        self.build()
        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                          '// Break here', self.main_source_spec)
        self.assertEquals(process.GetState(), lldb.eStateStopped)
        # FIXME: This pulls in the debug information to make the completions work,
        # but the completions should also work without.
        self.runCmd("frame variable fooo")

        self.complete_from_to('frame variable fo',
                              'frame variable fooo')
        self.complete_from_to('frame variable fooo.',
                              'frame variable fooo.')
        self.complete_from_to('frame variable fooo.dd',
                              'frame variable fooo.dd')

        self.complete_from_to('frame variable ptr_fooo->',
                              'frame variable ptr_fooo->')
        self.complete_from_to('frame variable ptr_fooo->dd',
                              'frame variable ptr_fooo->dd')

        self.complete_from_to('frame variable cont',
                              'frame variable container')
        self.complete_from_to('frame variable container.',
                              'frame variable container.MemberVar')
        self.complete_from_to('frame variable container.Mem',
                              'frame variable container.MemberVar')

        self.complete_from_to('frame variable ptr_cont',
                              'frame variable ptr_container')
        self.complete_from_to('frame variable ptr_container->',
                              'frame variable ptr_container->MemberVar')
        self.complete_from_to('frame variable ptr_container->Mem',
                              'frame variable ptr_container->MemberVar')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_process_attach_dash_dash_con(self):
        """Test that 'process attach --con' completes to 'process attach --continue '."""
        self.complete_from_to(
            'process attach --con',
            'process attach --continue ')

    # <rdar://problem/11052829>
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_infinite_loop_while_completing(self):
        """Test that 'process print hello\' completes to itself and does not infinite loop."""
        self.complete_from_to('process print hello\\', 'process print hello\\',
                              turn_off_re_match=True)

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_watchpoint_co(self):
        """Test that 'watchpoint co' completes to 'watchpoint command '."""
        self.complete_from_to('watchpoint co', 'watchpoint command ')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_watchpoint_command_space(self):
        """Test that 'watchpoint command ' completes to ['add', 'delete', 'list']."""
        self.complete_from_to(
            'watchpoint command ', [
                'add', 'delete', 'list'])

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_watchpoint_command_a(self):
        """Test that 'watchpoint command a' completes to 'watchpoint command add '."""
        self.complete_from_to(
            'watchpoint command a',
            'watchpoint command add ')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_watchpoint_set_ex(self):
        """Test that 'watchpoint set ex' completes to 'watchpoint set expression '."""
        self.complete_from_to(
            'watchpoint set ex',
            'watchpoint set expression ')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_watchpoint_set_var(self):
        """Test that 'watchpoint set var' completes to 'watchpoint set variable '."""
        self.complete_from_to('watchpoint set var', 'watchpoint set variable ')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_help_fi(self):
        """Test that 'help fi' completes to ['file', 'finish']."""
        self.complete_from_to(
            'help fi', [
                'file', 'finish'])

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_help_watchpoint_s(self):
        """Test that 'help watchpoint s' completes to 'help watchpoint set '."""
        self.complete_from_to('help watchpoint s', 'help watchpoint set ')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_append_target_er(self):
        """Test that 'settings append target.er' completes to 'settings append target.error-path'."""
        self.complete_from_to(
            'settings append target.er',
            'settings append target.error-path')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_insert_after_target_en(self):
        """Test that 'settings insert-after target.env' completes to 'settings insert-after target.env-vars'."""
        self.complete_from_to(
            'settings insert-after target.env',
            'settings insert-after target.env-vars')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_insert_before_target_en(self):
        """Test that 'settings insert-before target.env' completes to 'settings insert-before target.env-vars'."""
        self.complete_from_to(
            'settings insert-before target.env',
            'settings insert-before target.env-vars')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_replace_target_ru(self):
        """Test that 'settings replace target.ru' completes to 'settings replace target.run-args'."""
        self.complete_from_to(
            'settings replace target.ru',
            'settings replace target.run-args')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_s(self):
        """Test that 'settings s' completes to ['set', 'show']."""
        self.complete_from_to(
            'settings s', [
                'set', 'show'])

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_set_th(self):
        """Test that 'settings set thread-f' completes to 'settings set thread-format'."""
        self.complete_from_to('settings set thread-f', 'settings set thread-format')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_s_dash(self):
        """Test that 'settings set -' completes to 'settings set -g'."""
        self.complete_from_to('settings set -', 'settings set -g')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_clear_th(self):
        """Test that 'settings clear thread-f' completes to 'settings clear thread-format'."""
        self.complete_from_to(
            'settings clear thread-f',
            'settings clear thread-format')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_set_ta(self):
        """Test that 'settings set ta' completes to 'settings set target.'."""
        self.complete_from_to(
            'settings set target.ma',
            'settings set target.max-')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_set_target_exec(self):
        """Test that 'settings set target.exec' completes to 'settings set target.exec-search-paths '."""
        self.complete_from_to(
            'settings set target.exec',
            'settings set target.exec-search-paths')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_set_target_pr(self):
        """Test that 'settings set target.pr' completes to [
        'target.prefer-dynamic-value', 'target.process.']."""
        self.complete_from_to('settings set target.pr',
                              ['target.prefer-dynamic-value',
                               'target.process.'])

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_set_target_process(self):
        """Test that 'settings set target.process' completes to 'settings set target.process.'."""
        self.complete_from_to(
            'settings set target.process',
            'settings set target.process.')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_set_target_process_dot(self):
        """Test that 'settings set target.process.t' completes to 'settings set target.process.thread.'."""
        self.complete_from_to(
            'settings set target.process.t',
            'settings set target.process.thread.')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_settings_set_target_process_thread_dot(self):
        """Test that 'settings set target.process.thread.' completes to [
        'target.process.thread.step-avoid-regexp', 'target.process.thread.trace-thread']."""
        self.complete_from_to('settings set target.process.thread.',
                              ['target.process.thread.step-avoid-regexp',
                               'target.process.thread.trace-thread'])

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_target_space(self):
        """Test that 'target ' completes to ['create', 'delete', 'list',
        'modules', 'select', 'stop-hook', 'variable']."""
        self.complete_from_to('target ',
                              ['create',
                               'delete',
                               'list',
                               'modules',
                               'select',
                               'stop-hook',
                               'variable'])

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_target_create_dash_co(self):
        """Test that 'target create --co' completes to 'target variable --core '."""
        self.complete_from_to('target create --co', 'target create --core ')

    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    def test_target_va(self):
        """Test that 'target va' completes to 'target variable '."""
        self.complete_from_to('target va', 'target variable ')

    def test_command_argument_completion(self):
        """Test completion of command arguments"""
        self.complete_from_to("watchpoint set variable -", ["-w", "-s"])
        self.complete_from_to('watchpoint set variable -w', 'watchpoint set variable -w ')
        self.complete_from_to("watchpoint set variable --", ["--watch", "--size"])
        self.complete_from_to("watchpoint set variable --w", "watchpoint set variable --watch")
        self.complete_from_to('watchpoint set variable -w ', ['read', 'write', 'read_write'])
        self.complete_from_to("watchpoint set variable --watch ", ["read", "write", "read_write"])
        self.complete_from_to("watchpoint set variable --watch w", "watchpoint set variable --watch write")
        self.complete_from_to('watchpoint set variable -w read_', 'watchpoint set variable -w read_write')
        # Now try the same thing with a variable name (non-option argument) to
        # test that getopts arg reshuffling doesn't confuse us.
        self.complete_from_to("watchpoint set variable foo -", ["-w", "-s"])
        self.complete_from_to('watchpoint set variable foo -w', 'watchpoint set variable foo -w ')
        self.complete_from_to("watchpoint set variable foo --", ["--watch", "--size"])
        self.complete_from_to("watchpoint set variable foo --w", "watchpoint set variable foo --watch")
        self.complete_from_to('watchpoint set variable foo -w ', ['read', 'write', 'read_write'])
        self.complete_from_to("watchpoint set variable foo --watch ", ["read", "write", "read_write"])
        self.complete_from_to("watchpoint set variable foo --watch w", "watchpoint set variable foo --watch write")
        self.complete_from_to('watchpoint set variable foo -w read_', 'watchpoint set variable foo -w read_write')

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24489")
    def test_symbol_name(self):
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.complete_from_to('breakpoint set -n Fo',
                              'breakpoint set -n Foo::Bar(int,\\ int)',
                              turn_off_re_match=True)

    def complete_from_to(self, str_input, patterns, turn_off_re_match=False):
        """Test that the completion mechanism completes str_input to patterns,
        where patterns could be a pattern-string or a list of pattern-strings"""
        # Patterns should not be None in order to proceed.
        self.assertFalse(patterns is None)
        # And should be either a string or list of strings.  Check for list type
        # below, if not, make a list out of the singleton string.  If patterns
        # is not a string or not a list of strings, there'll be runtime errors
        # later on.
        if not isinstance(patterns, list):
            patterns = [patterns]

        interp = self.dbg.GetCommandInterpreter()
        match_strings = lldb.SBStringList()
        num_matches = interp.HandleCompletion(str_input, len(str_input), 0, -1, match_strings)
        common_match = match_strings.GetStringAtIndex(0)
        if num_matches == 0:
            compare_string = str_input
        else:
            if common_match != None and len(common_match) > 0:
                compare_string = str_input + common_match
            else:
                compare_string = ""
                for idx in range(1, num_matches+1):
                    compare_string += match_strings.GetStringAtIndex(idx) + "\n"

        for p in patterns:
            if turn_off_re_match:
                self.expect(
                    compare_string, msg=COMPLETION_MSG(
                        str_input, p, match_strings), exe=False, substrs=[p])
            else:
                self.expect(
                    compare_string, msg=COMPLETION_MSG(
                        str_input, p, match_strings), exe=False, patterns=[p])
