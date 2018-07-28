"""
Test lldb settings command.
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SettingsCommandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test_apropos_should_also_search_settings_description(self):
        """Test that 'apropos' command should also search descriptions for the settings variables."""

        self.expect("apropos 'environment variable'",
                    substrs=["target.env-vars",
                             "environment variables",
                             "executable's environment"])

    def test_append_target_env_vars(self):
        """Test that 'append target.run-args' works."""
        # Append the env-vars.
        self.runCmd('settings append target.env-vars MY_ENV_VAR=YES')
        # And add hooks to restore the settings during tearDown().
        self.addTearDownHook(
            lambda: self.runCmd("settings clear target.env-vars"))

        # Check it immediately!
        self.expect('settings show target.env-vars',
                    substrs=['MY_ENV_VAR=YES'])

    def test_insert_before_and_after_target_run_args(self):
        """Test that 'insert-before/after target.run-args' works."""
        # Set the run-args first.
        self.runCmd('settings set target.run-args a b c')
        # And add hooks to restore the settings during tearDown().
        self.addTearDownHook(
            lambda: self.runCmd("settings clear target.run-args"))

        # Now insert-before the index-0 element with '__a__'.
        self.runCmd('settings insert-before target.run-args 0 __a__')
        # And insert-after the index-1 element with '__A__'.
        self.runCmd('settings insert-after target.run-args 1 __A__')
        # Check it immediately!
        self.expect('settings show target.run-args',
                    substrs=['target.run-args',
                             '[0]: "__a__"',
                             '[1]: "a"',
                             '[2]: "__A__"',
                             '[3]: "b"',
                             '[4]: "c"'])

    def test_replace_target_run_args(self):
        """Test that 'replace target.run-args' works."""
        # Set the run-args and then replace the index-0 element.
        self.runCmd('settings set target.run-args a b c')
        # And add hooks to restore the settings during tearDown().
        self.addTearDownHook(
            lambda: self.runCmd("settings clear target.run-args"))

        # Now replace the index-0 element with 'A', instead.
        self.runCmd('settings replace target.run-args 0 A')
        # Check it immediately!
        self.expect('settings show target.run-args',
                    substrs=['target.run-args (arguments) =',
                             '[0]: "A"',
                             '[1]: "b"',
                             '[2]: "c"'])

    def test_set_prompt(self):
        """Test that 'set prompt' actually changes the prompt."""

        # Set prompt to 'lldb2'.
        self.runCmd("settings set prompt 'lldb2 '")

        # Immediately test the setting.
        self.expect("settings show prompt", SETTING_MSG("prompt"),
                    startstr='prompt (string) = "lldb2 "')

        # The overall display should also reflect the new setting.
        self.expect("settings show", SETTING_MSG("prompt"),
                    substrs=['prompt (string) = "lldb2 "'])

        # Use '-r' option to reset to the original default prompt.
        self.runCmd("settings clear prompt")

    def test_set_term_width(self):
        """Test that 'set term-width' actually changes the term-width."""

        self.runCmd("settings set term-width 70")

        # Immediately test the setting.
        self.expect("settings show term-width", SETTING_MSG("term-width"),
                    startstr="term-width (int) = 70")

        # The overall display should also reflect the new setting.
        self.expect("settings show", SETTING_MSG("term-width"),
                    substrs=["term-width (int) = 70"])

    # rdar://problem/10712130
    def test_set_frame_format(self):
        """Test that 'set frame-format' with a backtick char in the format string works as well as fullpath."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        def cleanup():
            self.runCmd(
                "settings set frame-format %s" %
                self.format_string, check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("settings show frame-format")
        m = re.match(
            '^frame-format \(format-string\) = "(.*)\"$',
            self.res.GetOutput())
        self.assertTrue(m, "Bad settings string")
        self.format_string = m.group(1)

        # Change the default format to print function.name rather than
        # function.name-with-args
        format_string = "frame #${frame.index}: ${frame.pc}{ ${module.file.basename}`${function.name}{${function.pc-offset}}}{ at ${line.file.fullpath}:${line.number}}{, lang=${language}}\n"
        self.runCmd("settings set frame-format %s" % format_string)

        # Immediately test the setting.
        self.expect("settings show frame-format", SETTING_MSG("frame-format"),
                    substrs=[format_string])

        self.runCmd("breakpoint set -n main")
        self.runCmd("process launch --working-dir '{0}'".format(self.get_process_working_directory()),
                RUN_SUCCEEDED)
        self.expect("thread backtrace",
                    substrs=["`main", self.getSourceDir()])

    def test_set_auto_confirm(self):
        """Test that after 'set auto-confirm true', manual confirmation should not kick in."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("settings set auto-confirm true")

        # Immediately test the setting.
        self.expect("settings show auto-confirm", SETTING_MSG("auto-confirm"),
                    startstr="auto-confirm (boolean) = true")

        # Now 'breakpoint delete' should just work fine without confirmation
        # prompt from the command interpreter.
        self.runCmd("breakpoint set -n main")
        self.expect("breakpoint delete",
                    startstr="All breakpoints removed")

        # Restore the original setting of auto-confirm.
        self.runCmd("settings clear auto-confirm")
        self.expect("settings show auto-confirm", SETTING_MSG("auto-confirm"),
                    startstr="auto-confirm (boolean) = false")

    @skipIf(archs=no_match(['x86_64', 'i386', 'i686']))
    def test_disassembler_settings(self):
        """Test that user options for the disassembler take effect."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # AT&T syntax
        self.runCmd("settings set target.x86-disassembly-flavor att")
        self.runCmd("settings set target.use-hex-immediates false")
        self.expect("disassemble -n numberfn",
                    substrs=["$90"])
        self.runCmd("settings set target.use-hex-immediates true")
        self.runCmd("settings set target.hex-immediate-style c")
        self.expect("disassemble -n numberfn",
                    substrs=["$0x5a"])
        self.runCmd("settings set target.hex-immediate-style asm")
        self.expect("disassemble -n numberfn",
                    substrs=["$5ah"])

        # Intel syntax
        self.runCmd("settings set target.x86-disassembly-flavor intel")
        self.runCmd("settings set target.use-hex-immediates false")
        self.expect("disassemble -n numberfn",
                    substrs=["90"])
        self.runCmd("settings set target.use-hex-immediates true")
        self.runCmd("settings set target.hex-immediate-style c")
        self.expect("disassemble -n numberfn",
                    substrs=["0x5a"])
        self.runCmd("settings set target.hex-immediate-style asm")
        self.expect("disassemble -n numberfn",
                    substrs=["5ah"])

    @skipIfDarwinEmbedded   # <rdar://problem/34446098> debugserver on ios etc can't write files
    def test_run_args_and_env_vars(self):
        """Test that run-args and env-vars are passed to the launched process."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set the run-args and the env-vars.
        # And add hooks to restore the settings during tearDown().
        self.runCmd('settings set target.run-args A B C')
        self.addTearDownHook(
            lambda: self.runCmd("settings clear target.run-args"))
        self.runCmd('settings set target.env-vars ["MY_ENV_VAR"]=YES')
        self.addTearDownHook(
            lambda: self.runCmd("settings clear target.env-vars"))

        self.runCmd("process launch --working-dir '{0}'".format(self.get_process_working_directory()),
                RUN_SUCCEEDED)

        # Read the output file produced by running the program.
        output = lldbutil.read_file_from_process_wd(self, "output2.txt")

        self.expect(
            output,
            exe=False,
            substrs=[
                "argv[1] matches",
                "argv[2] matches",
                "argv[3] matches",
                "Environment variable 'MY_ENV_VAR' successfully passed."])

    @skipIfRemote  # it doesn't make sense to send host env to remote target
    def test_pass_host_env_vars(self):
        """Test that the host env vars are passed to the launched process."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # By default, inherit-env is 'true'.
        self.expect(
            'settings show target.inherit-env',
            "Default inherit-env is 'true'",
            startstr="target.inherit-env (boolean) = true")

        # Set some host environment variables now.
        os.environ["MY_HOST_ENV_VAR1"] = "VAR1"
        os.environ["MY_HOST_ENV_VAR2"] = "VAR2"

        # This is the function to unset the two env variables set above.
        def unset_env_variables():
            os.environ.pop("MY_HOST_ENV_VAR1")
            os.environ.pop("MY_HOST_ENV_VAR2")

        self.addTearDownHook(unset_env_variables)
        self.runCmd("process launch --working-dir '{0}'".format(self.get_process_working_directory()),
                RUN_SUCCEEDED)

        # Read the output file produced by running the program.
        output = lldbutil.read_file_from_process_wd(self, "output1.txt")

        self.expect(
            output,
            exe=False,
            substrs=[
                "The host environment variable 'MY_HOST_ENV_VAR1' successfully passed.",
                "The host environment variable 'MY_HOST_ENV_VAR2' successfully passed."])

    @skipIfDarwinEmbedded   # <rdar://problem/34446098> debugserver on ios etc can't write files
    def test_set_error_output_path(self):
        """Test that setting target.error/output-path for the launched process works."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set the error-path and output-path and verify both are set.
        self.runCmd("settings set target.error-path '{0}'".format(
            lldbutil.append_to_process_working_directory(self, "stderr.txt")))
        self.runCmd("settings set target.output-path '{0}".format(
            lldbutil.append_to_process_working_directory(self, "stdout.txt")))
        # And add hooks to restore the original settings during tearDown().
        self.addTearDownHook(
            lambda: self.runCmd("settings clear target.output-path"))
        self.addTearDownHook(
            lambda: self.runCmd("settings clear target.error-path"))

        self.expect("settings show target.error-path",
                    SETTING_MSG("target.error-path"),
                    substrs=['target.error-path (file)', 'stderr.txt"'])

        self.expect("settings show target.output-path",
                    SETTING_MSG("target.output-path"),
                    substrs=['target.output-path (file)', 'stdout.txt"'])

        self.runCmd("process launch --working-dir '{0}'".format(self.get_process_working_directory()),
                RUN_SUCCEEDED)

        output = lldbutil.read_file_from_process_wd(self, "stderr.txt")
        message = "This message should go to standard error."
        if lldbplatformutil.hasChattyStderr(self):
            self.expect(output, exe=False, substrs=[message])
        else:
            self.expect(output, exe=False, startstr=message)

        output = lldbutil.read_file_from_process_wd(self, "stdout.txt")
        self.expect(output, exe=False,
                    startstr="This message should go to standard out.")

    def test_print_dictionary_setting(self):
        self.runCmd("settings clear target.env-vars")
        self.runCmd("settings set target.env-vars [\"MY_VAR\"]=some-value")
        self.expect("settings show target.env-vars",
                    substrs=["MY_VAR=some-value"])
        self.runCmd("settings clear target.env-vars")

    def test_print_array_setting(self):
        self.runCmd("settings clear target.run-args")
        self.runCmd("settings set target.run-args gobbledy-gook")
        self.expect("settings show target.run-args",
                    substrs=['[0]: "gobbledy-gook"'])
        self.runCmd("settings clear target.run-args")

    def test_settings_with_quotes(self):
        self.runCmd("settings clear target.run-args")
        self.runCmd("settings set target.run-args a b c")
        self.expect("settings show target.run-args",
                    substrs=['[0]: "a"',
                             '[1]: "b"',
                             '[2]: "c"'])
        self.runCmd("settings set target.run-args 'a b c'")
        self.expect("settings show target.run-args",
                    substrs=['[0]: "a b c"'])
        self.runCmd("settings clear target.run-args")
        self.runCmd("settings clear target.env-vars")
        self.runCmd(
            'settings set target.env-vars ["MY_FILE"]="this is a file name with spaces.txt"')
        self.expect("settings show target.env-vars",
                    substrs=['MY_FILE=this is a file name with spaces.txt'])
        self.runCmd("settings clear target.env-vars")
        # Test and make sure that setting "format-string" settings obeys quotes
        # if they are provided
        self.runCmd("settings set thread-format    'abc def'   ")
        self.expect("settings show thread-format",
                    'thread-format (format-string) = "abc def"')
        self.runCmd('settings set thread-format    "abc def"   ')
        self.expect("settings show thread-format",
                    'thread-format (format-string) = "abc def"')
        # Make sure when no quotes are provided that we maintain any trailing
        # spaces
        self.runCmd('settings set thread-format abc def   ')
        self.expect("settings show thread-format",
                    'thread-format (format-string) = "abc def   "')
        self.runCmd('settings clear thread-format')

    def test_settings_with_trailing_whitespace(self):

        # boolean
        # Set to known value
        self.runCmd("settings set target.skip-prologue true")
        # Set to new value with trailing whitespace
        self.runCmd("settings set target.skip-prologue false ")
        # Make sure the setting was correctly set to "false"
        self.expect(
            "settings show target.skip-prologue",
            SETTING_MSG("target.skip-prologue"),
            startstr="target.skip-prologue (boolean) = false")
        self.runCmd("settings clear target.skip-prologue", check=False)
        # integer
        self.runCmd("settings set term-width 70")      # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd("settings set term-width 60 \t")
        self.expect("settings show term-width", SETTING_MSG("term-width"),
                    startstr="term-width (int) = 60")
        self.runCmd("settings clear term-width", check=False)
        # string
        self.runCmd("settings set target.arg0 abc")    # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd("settings set target.arg0 cde\t ")
        self.expect("settings show target.arg0", SETTING_MSG("target.arg0"),
                    startstr='target.arg0 (string) = "cde"')
        self.runCmd("settings clear target.arg0", check=False)
        # file
        path1 = self.getBuildArtifact("path1.txt")
        path2 = self.getBuildArtifact("path2.txt")
        self.runCmd(
            "settings set target.output-path %s" %
            path1)   # Set to known value
        self.expect(
            "settings show target.output-path",
            SETTING_MSG("target.output-path"),
            startstr='target.output-path (file) = ',
            substrs=[path1])
        self.runCmd("settings set target.output-path %s " %
                    path2)  # Set to new value with trailing whitespaces
        self.expect(
            "settings show target.output-path",
            SETTING_MSG("target.output-path"),
            startstr='target.output-path (file) = ',
            substrs=[path2])
        self.runCmd("settings clear target.output-path", check=False)
        # enum
        # Set to known value
        self.runCmd("settings set stop-disassembly-display never")
        # Set to new value with trailing whitespaces
        self.runCmd("settings set stop-disassembly-display always ")
        self.expect(
            "settings show stop-disassembly-display",
            SETTING_MSG("stop-disassembly-display"),
            startstr='stop-disassembly-display (enum) = always')
        self.runCmd("settings clear stop-disassembly-display", check=False)
        # language
        # Set to known value
        self.runCmd("settings set target.language c89")
        # Set to new value with trailing whitespace
        self.runCmd("settings set target.language go ")
        self.expect(
            "settings show target.language",
            SETTING_MSG("target.language"),
            startstr="target.language (language) = go")
        self.runCmd("settings clear target.language", check=False)
        # arguments
        self.runCmd("settings set target.run-args 1 2 3")  # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd("settings set target.run-args 3 4 5 ")
        self.expect(
            "settings show target.run-args",
            SETTING_MSG("target.run-args"),
            substrs=[
                'target.run-args (arguments) =',
                '[0]: "3"',
                '[1]: "4"',
                '[2]: "5"'])
        self.runCmd("settings set target.run-args 1 2 3")  # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd("settings set target.run-args 3 \  \ ")
        self.expect(
            "settings show target.run-args",
            SETTING_MSG("target.run-args"),
            substrs=[
                'target.run-args (arguments) =',
                '[0]: "3"',
                '[1]: " "',
                '[2]: " "'])
        self.runCmd("settings clear target.run-args", check=False)
        # dictionaries
        self.runCmd("settings clear target.env-vars")  # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd("settings set target.env-vars A=B C=D\t ")
        self.expect(
            "settings show target.env-vars",
            SETTING_MSG("target.env-vars"),
            substrs=[
                'target.env-vars (dictionary of strings) =',
                'A=B',
                'C=D'])
        self.runCmd("settings clear target.env-vars", check=False)
        # regex
        # Set to known value
        self.runCmd("settings clear target.process.thread.step-avoid-regexp")
        # Set to new value with trailing whitespaces
        self.runCmd(
            "settings set target.process.thread.step-avoid-regexp foo\\ ")
        self.expect(
            "settings show target.process.thread.step-avoid-regexp",
            SETTING_MSG("target.process.thread.step-avoid-regexp"),
            substrs=['target.process.thread.step-avoid-regexp (regex) = foo\\ '])
        self.runCmd(
            "settings clear target.process.thread.step-avoid-regexp",
            check=False)
        # format-string
        self.runCmd("settings clear disassembly-format")  # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd("settings set disassembly-format foo ")
        self.expect("settings show disassembly-format",
                    SETTING_MSG("disassembly-format"),
                    substrs=['disassembly-format (format-string) = "foo "'])
        self.runCmd("settings clear disassembly-format", check=False)

    def test_all_settings_exist(self):
        self.expect("settings show",
                    substrs=["auto-confirm",
                             "frame-format",
                             "notify-void",
                             "prompt",
                             "script-lang",
                             "stop-disassembly-count",
                             "stop-disassembly-display",
                             "stop-line-count-after",
                             "stop-line-count-before",
                             "stop-show-column",
                             "term-width",
                             "thread-format",
                             "use-external-editor",
                             "target.default-arch",
                             "target.move-to-nearest-code",
                             "target.expr-prefix",
                             "target.language",
                             "target.prefer-dynamic-value",
                             "target.enable-synthetic-value",
                             "target.skip-prologue",
                             "target.source-map",
                             "target.exec-search-paths",
                             "target.max-children-count",
                             "target.max-string-summary-length",
                             "target.breakpoints-use-platform-avoid-list",
                             "target.run-args",
                             "target.env-vars",
                             "target.inherit-env",
                             "target.input-path",
                             "target.output-path",
                             "target.error-path",
                             "target.disable-aslr",
                             "target.disable-stdio",
                             "target.x86-disassembly-flavor",
                             "target.use-hex-immediates",
                             "target.hex-immediate-style",
                             "target.process.disable-memory-cache",
                             "target.process.extra-startup-command",
                             "target.process.thread.step-avoid-regexp",
                             "target.process.thread.trace-thread"])

    # settings under an ".experimental" domain should have two properties:
    #   1. If the name does not exist with "experimental" in the name path,
    #      the name lookup should try to find it without "experimental".  So
    #      a previously-experimental setting that has been promoted to a
    #      "real" setting will still be set by the original name.
    #   2. Changing a setting with .experimental., name, where the setting
    #      does not exist either with ".experimental." or without, should
    #      not generate an error.  So if an experimental setting is removed,
    #      people who may have that in their ~/.lldbinit files should not see
    #      any errors.
    def test_experimental_settings(self):
        cmdinterp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        # Set target.arg0 to a known value, check that we can retrieve it via
        # the actual name and via .experimental.
        self.expect('settings set target.arg0 first-value')
        self.expect('settings show target.arg0', substrs=['first-value'])
        self.expect('settings show target.experimental.arg0', substrs=['first-value'], error=False)

        # Set target.arg0 to a new value via a target.experimental.arg0 name,
        # verify that we can read it back via both .experimental., and not.
        self.expect('settings set target.experimental.arg0 second-value', error=False)
        self.expect('settings show target.arg0', substrs=['second-value'])
        self.expect('settings show target.experimental.arg0', substrs=['second-value'], error=False)

        # showing & setting an undefined .experimental. setting should generate no errors.
        self.expect('settings show target.experimental.setting-which-does-not-exist', patterns=['^\s$'], error=False)
        self.expect('settings set target.experimental.setting-which-does-not-exist true', error=False)

        # A domain component before .experimental. which does not exist should give an error
        # But the code does not yet do that.
        # self.expect('settings set target.setting-which-does-not-exist.experimental.arg0 true', error=True)

        # finally, confirm that trying to set a setting that does not exist still fails.
        # (SHOWING a setting that does not exist does not currently yield an error.)
        self.expect('settings set target.setting-which-does-not-exist true', error=True)
