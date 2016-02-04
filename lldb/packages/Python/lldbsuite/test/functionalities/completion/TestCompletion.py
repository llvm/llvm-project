"""
Test the lldb command line completion mechanism.
"""

from __future__ import print_function



import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class CommandLineCompletionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproducts."""
        try:
            os.remove("child_send.txt")
            os.remove("child_read.txt")
        except:
            pass

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_at(self):
        """Test that 'at' completes to 'attach '."""
        self.complete_from_to('at', 'attach ')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_de(self):
        """Test that 'de' completes to 'detach '."""
        self.complete_from_to('de', 'detach ')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_process_attach_dash_dash_con(self):
        """Test that 'process attach --con' completes to 'process attach --continue '."""
        self.complete_from_to('process attach --con', 'process attach --continue ')

    # <rdar://problem/11052829>
    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_infinite_loop_while_completing(self):
        """Test that 'process print hello\' completes to itself and does not infinite loop."""
        self.complete_from_to('process print hello\\', 'process print hello\\',
                              turn_off_re_match=True)

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_co(self):
        """Test that 'watchpoint co' completes to 'watchpoint command '."""
        self.complete_from_to('watchpoint co', 'watchpoint command ')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_command_space(self):
        """Test that 'watchpoint command ' completes to ['Available completions:', 'add', 'delete', 'list']."""
        self.complete_from_to('watchpoint command ', ['Available completions:', 'add', 'delete', 'list'])

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_command_a(self):
        """Test that 'watchpoint command a' completes to 'watchpoint command add '."""
        self.complete_from_to('watchpoint command a', 'watchpoint command add ')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_set_variable_dash_w(self):
        """Test that 'watchpoint set variable -w' completes to 'watchpoint set variable -w '."""
        self.complete_from_to('watchpoint set variable -w', 'watchpoint set variable -w ')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_set_variable_dash_w_space(self):
        """Test that 'watchpoint set variable -w ' completes to ['Available completions:', 'read', 'write', 'read_write']."""
        self.complete_from_to('watchpoint set variable -w ', ['Available completions:', 'read', 'write', 'read_write'])

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_set_ex(self):
        """Test that 'watchpoint set ex' completes to 'watchpoint set expression '."""
        self.complete_from_to('watchpoint set ex', 'watchpoint set expression ')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_set_var(self):
        """Test that 'watchpoint set var' completes to 'watchpoint set variable '."""
        self.complete_from_to('watchpoint set var', 'watchpoint set variable ')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_watchpoint_set_variable_dash_w_read_underbar(self):
        """Test that 'watchpoint set variable -w read_' completes to 'watchpoint set variable -w read_write'."""
        self.complete_from_to('watchpoint set variable -w read_', 'watchpoint set variable -w read_write')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_help_fi(self):
        """Test that 'help fi' completes to ['Available completions:', 'file', 'finish']."""
        self.complete_from_to('help fi', ['Available completions:', 'file', 'finish'])

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_help_watchpoint_s(self):
        """Test that 'help watchpoint s' completes to 'help watchpoint set '."""
        self.complete_from_to('help watchpoint s', 'help watchpoint set ')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_append_target_er(self):
        """Test that 'settings append target.er' completes to 'settings append target.error-path'."""
        self.complete_from_to('settings append target.er', 'settings append target.error-path')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_insert_after_target_en(self):
        """Test that 'settings insert-after target.env' completes to 'settings insert-after target.env-vars'."""
        self.complete_from_to('settings insert-after target.env', 'settings insert-after target.env-vars')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_insert_before_target_en(self):
        """Test that 'settings insert-before target.env' completes to 'settings insert-before target.env-vars'."""
        self.complete_from_to('settings insert-before target.env', 'settings insert-before target.env-vars')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_replace_target_ru(self):
        """Test that 'settings replace target.ru' completes to 'settings replace target.run-args'."""
        self.complete_from_to('settings replace target.ru', 'settings replace target.run-args')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_s(self):
        """Test that 'settings s' completes to ['Available completions:', 'set', 'show']."""
        self.complete_from_to('settings s', ['Available completions:', 'set', 'show'])

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_th(self):
        """Test that 'settings set th' completes to 'settings set thread-format'."""
        self.complete_from_to('settings set th', 'settings set thread-format')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_s_dash(self):
        """Test that 'settings set -' completes to 'settings set -g'."""
        self.complete_from_to('settings set -', 'settings set -g')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_clear_th(self):
        """Test that 'settings clear th' completes to 'settings clear thread-format'."""
        self.complete_from_to('settings clear th', 'settings clear thread-format')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_ta(self):
        """Test that 'settings set ta' completes to 'settings set target.'."""
        self.complete_from_to('settings set target.ma', 'settings set target.max-')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_target_exec(self):
        """Test that 'settings set target.exec' completes to 'settings set target.exec-search-paths '."""
        self.complete_from_to('settings set target.exec', 'settings set target.exec-search-paths')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_target_pr(self):
        """Test that 'settings set target.pr' completes to ['Available completions:',
        'target.prefer-dynamic-value', 'target.process.']."""
        self.complete_from_to('settings set target.pr',
                              ['Available completions:',
                               'target.prefer-dynamic-value',
                               'target.process.'])

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_target_process(self):
        """Test that 'settings set target.process' completes to 'settings set target.process.'."""
        self.complete_from_to('settings set target.process', 'settings set target.process.')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_target_process_dot(self):
        """Test that 'settings set target.process.t' completes to 'settings set target.process.thread.'."""
        self.complete_from_to('settings set target.process.t', 'settings set target.process.thread.')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_settings_set_target_process_thread_dot(self):
        """Test that 'settings set target.process.thread.' completes to ['Available completions:',
        'target.process.thread.step-avoid-regexp', 'target.process.thread.trace-thread']."""
        self.complete_from_to('settings set target.process.thread.',
                              ['Available completions:',
                               'target.process.thread.step-avoid-regexp',
                               'target.process.thread.trace-thread'])

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_target_space(self):
        """Test that 'target ' completes to ['Available completions:', 'create', 'delete', 'list',
        'modules', 'select', 'stop-hook', 'variable']."""
        self.complete_from_to('target ',
                              ['Available completions:', 'create', 'delete', 'list',
                               'modules', 'select', 'stop-hook', 'variable'])

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_target_create_dash_co(self):
        """Test that 'target create --co' completes to 'target variable --core '."""
        self.complete_from_to('target create --co', 'target create --core ')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @skipIfFreeBSD # timing out on the FreeBSD buildbot
    @no_debug_info_test
    def test_target_va(self):
        """Test that 'target va' completes to 'target variable '."""
        self.complete_from_to('target va', 'target variable ')

    @expectedFailureHostWindows("llvm.org/pr24679")
    @expectedFailureDarwin("llvm.org/pr25485")
    def test_symbol_name(self):
        self.build()
        self.complete_from_to('''file a.out
                                 breakpoint set -n Fo''',
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
        
        # The default lldb prompt.
        prompt = "(lldb) "

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn(lldbtest_config.lldbExec,
                                   [self.lldbOption] + ['--no-use-colors'])
        child = self.child
        # Turn on logging for input/output to/from the child.
        with open('child_send.txt', 'w') as f_send:
            with open('child_read.txt', 'w') as f_read:
                child.logfile_send = f_send
                child.logfile_read = f_read

                child.expect_exact(prompt)
                child.setecho(True)
                # Sends str_input and a Tab to invoke the completion machinery.
                child.send("%s\t" % str_input)
                child.sendline('')
                child.expect_exact(prompt)
                child.sendline('')
                child.expect_exact(prompt)

        # Now that the necessary logging is done, restore logfile to None to
        # stop further logging.
        child.logfile_send = None
        child.logfile_read = None
        
        with open('child_send.txt', 'r') as fs:
            if self.TraceOn():
                print("\n\nContents of child_send.txt:")
                print(fs.read())
        with open('child_read.txt', 'r') as fr:
            from_child = fr.read()
            if self.TraceOn():
                print("\n\nContents of child_read.txt:")
                print(from_child)

            # The matching could be verbatim or using generic re pattern.
            for p in patterns:
                # Test that str_input completes to our patterns or substrings.
                # If each pattern/substring matches from_child, the completion mechanism works!
                if turn_off_re_match:
                    self.expect(from_child, msg=COMPLETION_MSG(str_input, p), exe=False,
                        substrs = [p])
                else:
                    self.expect(from_child, msg=COMPLETION_MSG(str_input, p), exe=False,
                        patterns = [p])
