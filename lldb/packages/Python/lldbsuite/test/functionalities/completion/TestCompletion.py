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

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_at(self):
        """Test that 'at' completes to 'attach '."""
        self.complete_from_to('at', 'attach ')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_de(self):
        """Test that 'de' completes to 'detach '."""
        self.complete_from_to('de', 'detach ')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_process_attach_dash_dash_con(self):
        """Test that 'process attach --con' completes to 'process attach --continue '."""
        self.complete_from_to(
            'process attach --con',
            'process attach --continue ')

    # <rdar://problem/11052829>
    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_infinite_loop_while_completing(self):
        """Test that 'process print hello\' completes to itself and does not infinite loop."""
        self.complete_from_to('process print hello\\', 'process print hello\\',
                              turn_off_re_match=True)

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_co(self):
        """Test that 'watchpoint co' completes to 'watchpoint command '."""
        self.complete_from_to('watchpoint co', 'watchpoint command ')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_command_space(self):
        """Test that 'watchpoint command ' completes to ['add', 'delete', 'list']."""
        self.complete_from_to(
            'watchpoint command ', [
                'add', 'delete', 'list'])

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_command_a(self):
        """Test that 'watchpoint command a' completes to 'watchpoint command add '."""
        self.complete_from_to(
            'watchpoint command a',
            'watchpoint command add ')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_set_variable_dash_w(self):
        """Test that 'watchpoint set variable -w' completes to 'watchpoint set variable -w '."""
        self.complete_from_to(
            'watchpoint set variable -w',
            'watchpoint set variable -w ')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_set_variable_dash_w_space(self):
        """Test that 'watchpoint set variable -w ' completes to ['read', 'write', 'read_write']."""
        self.complete_from_to('watchpoint set variable -w ',
                              ['read', 'write', 'read_write'])

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_set_ex(self):
        """Test that 'watchpoint set ex' completes to 'watchpoint set expression '."""
        self.complete_from_to(
            'watchpoint set ex',
            'watchpoint set expression ')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_set_var(self):
        """Test that 'watchpoint set var' completes to 'watchpoint set variable '."""
        self.complete_from_to('watchpoint set var', 'watchpoint set variable ')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_set_variable_dash_w_read_underbar(self):
        """Test that 'watchpoint set variable -w read_' completes to 'watchpoint set variable -w read_write'."""
        self.complete_from_to(
            'watchpoint set variable -w read_',
            'watchpoint set variable -w read_write')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_help_fi(self):
        """Test that 'help fi' completes to ['file', 'finish']."""
        self.complete_from_to(
            'help fi', [
                'file', 'finish'])

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_help_watchpoint_s(self):
        """Test that 'help watchpoint s' completes to 'help watchpoint set '."""
        self.complete_from_to('help watchpoint s', 'help watchpoint set ')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_append_target_er(self):
        """Test that 'settings append target.er' completes to 'settings append target.error-path'."""
        self.complete_from_to(
            'settings append target.er',
            'settings append target.error-path')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_insert_after_target_en(self):
        """Test that 'settings insert-after target.env' completes to 'settings insert-after target.env-vars'."""
        self.complete_from_to(
            'settings insert-after target.env',
            'settings insert-after target.env-vars')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_insert_before_target_en(self):
        """Test that 'settings insert-before target.env' completes to 'settings insert-before target.env-vars'."""
        self.complete_from_to(
            'settings insert-before target.env',
            'settings insert-before target.env-vars')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_replace_target_ru(self):
        """Test that 'settings replace target.ru' completes to 'settings replace target.run-args'."""
        self.complete_from_to(
            'settings replace target.ru',
            'settings replace target.run-args')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_s(self):
        """Test that 'settings s' completes to ['set', 'show']."""
        self.complete_from_to(
            'settings s', [
                'set', 'show'])

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_th(self):
        """Test that 'settings set thread-f' completes to 'settings set thread-format'."""
        self.complete_from_to('settings set thread-f', 'settings set thread-format')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_s_dash(self):
        """Test that 'settings set -' completes to 'settings set -g'."""
        self.complete_from_to('settings set -', 'settings set -g')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_clear_th(self):
        """Test that 'settings clear thread-f' completes to 'settings clear thread-format'."""
        self.complete_from_to(
            'settings clear thread-f',
            'settings clear thread-format')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_ta(self):
        """Test that 'settings set ta' completes to 'settings set target.'."""
        self.complete_from_to(
            'settings set target.ma',
            'settings set target.max-')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_target_exec(self):
        """Test that 'settings set target.exec' completes to 'settings set target.exec-search-paths '."""
        self.complete_from_to(
            'settings set target.exec',
            'settings set target.exec-search-paths')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_target_pr(self):
        """Test that 'settings set target.pr' completes to [
        'target.prefer-dynamic-value', 'target.process.']."""
        self.complete_from_to('settings set target.pr',
                              ['target.prefer-dynamic-value',
                               'target.process.'])

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_target_process(self):
        """Test that 'settings set target.process' completes to 'settings set target.process.'."""
        self.complete_from_to(
            'settings set target.process',
            'settings set target.process.')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_target_process_dot(self):
        """Test that 'settings set target.process.t' completes to 'settings set target.process.thread.'."""
        self.complete_from_to(
            'settings set target.process.t',
            'settings set target.process.thread.')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_target_process_thread_dot(self):
        """Test that 'settings set target.process.thread.' completes to [
        'target.process.thread.step-avoid-regexp', 'target.process.thread.trace-thread']."""
        self.complete_from_to('settings set target.process.thread.',
                              ['target.process.thread.step-avoid-regexp',
                               'target.process.thread.trace-thread'])

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
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

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_target_create_dash_co(self):
        """Test that 'target create --co' completes to 'target variable --core '."""
        self.complete_from_to('target create --co', 'target create --core ')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    @skipIfFreeBSD  # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_target_va(self):
        """Test that 'target va' completes to 'target variable '."""
        self.complete_from_to('target va', 'target variable ')

    @expectedFailureAll(hostoslist=["windows"], bugnumber="llvm.org/pr24679")
    def test_symbol_name(self):
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.complete_from_to('breakpoint set -n Fo',
                              'breakpoint set -n Foo::Bar(int,\\ int)',
                              turn_off_re_match=True)

    def complete_from_to(self, str_input, patterns, turn_off_re_match=False):
        """Test that the completion mechanism completes str_input to patterns,
        where patterns could be a pattern-string or a list of pattern-strings"""
        import pexpect
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
                        str_input, p), exe=False, substrs=[p])
            else:
                self.expect(
                    compare_string, msg=COMPLETION_MSG(
                        str_input, p), exe=False, patterns=[p])
