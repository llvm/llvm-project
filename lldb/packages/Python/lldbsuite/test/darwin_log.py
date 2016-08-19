"""
Base class for DarwinLog tests.
"""

# System imports
from __future__ import print_function

import json
import os
import pexpect
import platform
import re
import sys
import threading


# lldb imports
import lldb
from lldb import SBProcess, SBTarget

from lldbsuite.test import decorators
from lldbsuite.test import lldbtest
from lldbsuite.test import lldbtest_config
from lldbsuite.test import lldbutil


def expand_darwinlog_command(command):
    return "plugin structured-data darwin-log " + command


def expand_darwinlog_settings_set_command(command):
    return "settings set plugin.structured-data.darwin-log." + command


class DarwinLogTestBase(lldbtest.TestBase):
    """Base class for DarwinLog test cases that are pexpect-based."""
    NO_DEBUG_INFO_TESTCASE = True

    CONTINUE_REGEX = re.compile(r"Process \d+ resuming")

    def setUp(self):
        # Call super's setUp().
        super(DarwinLogTestBase, self).setUp()

        # Until other systems support this, exit
        # early if we're not macOS version 10.12
        # or greater.
        version = platform.mac_ver()[0].split('.')
        if ((int(version[0]) == 10) and (int(version[1]) < 12) or
            (int(version[0]) < 10)):
                self.skipTest("DarwinLog tests currently require macOS 10.12+")
                return

        self.child = None
        self.child_prompt = '(lldb) '
        self.strict_sources = False
        self.enable_process_monitor_logging = False

    def run_lldb_to_breakpoint(self, exe, source_file, line,
                               enable_command=None, settings_commands=None):

        # Set self.child_prompt, which is "(lldb) ".
        prompt = self.child_prompt

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s %s %s' % (lldbtest_config.lldbExec,
                                                 self.lldbOption, exe))
        child = self.child

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        if self.enable_process_monitor_logging:
            if platform.system() == 'Darwin':
                self.runCmd("settings set target.process.extra-startup-command "
                            "QSetLogging:bitmask=LOG_DARWIN_LOG;")
                self.expect_prompt()

        # Run the enable command if we have one.
        if enable_command is not None:
            self.runCmd(enable_command)
            self.expect_prompt()

        # Disable showing of source lines at our breakpoint.
        # This is necessary for the logging tests, because the very
        # text we want to match for output from the running inferior
        # will show up in the source as well.  We don't want the source
        # output to erroneously make a match with our expected output.
        self.runCmd("settings set stop-line-count-before 0")
        self.expect_prompt()
        self.runCmd("settings set stop-line-count-after 0")
        self.expect_prompt()

        # While we're debugging, turn on packet logging.
        self.runCmd("log enable -f /tmp/packets.log gdb-remote packets")
        self.expect_prompt()

        # Prevent mirroring of NSLog/os_log content to stderr.  We want log
        # messages to come exclusively through our log channel.
        self.runCmd("settings set target.env-vars IDE_DISABLED_OS_ACTIVITY_DT_MODE=1")
        self.expect_prompt()

        # Run any darwin-log settings commands now, before we enable logging.
        if settings_commands is not None:
            for setting_command in settings_commands:
                self.runCmd(
                    expand_darwinlog_settings_set_command(setting_command))
                self.expect_prompt()

        # Set breakpoint right before the os_log() macros.  We don't
        # set it on the os_log*() calls because these are a number of
        # nested-scoped calls that will cause the debugger to stop
        # multiple times on the same line.  That is difficult to match
        # os_log() content by since it is non-deterministic what the
        # ordering between stops and log lines will be.  This is why
        # we stop before, and then have the process run in a sleep
        # afterwards, so we get the log messages while the target
        # process is "running" (sleeping).
        child.sendline('breakpoint set -f %s -l %d' % (source_file, line))
        child.expect_exact(prompt)

        # Now run to the breakpoint that we just set.
        child.sendline('run')
        child.expect_exact(prompt)

        # Ensure we stopped at a breakpoint.
        self.runCmd("thread list")
        self.expect(re.compile(r"stop reason = breakpoint"))

        # Now we're ready to check if DarwinLog is available.
        if not self.darwin_log_available():
            self.skipTest("DarwinLog not available")

    def runCmd(self, cmd):
        self.child.sendline(cmd)

    def expect_prompt(self, exactly=True):
        self.expect(self.child_prompt, exactly=exactly)

    def expect(self, pattern, exactly=False, *args, **kwargs):
        if exactly:
            return self.child.expect_exact(pattern, *args, **kwargs)
        return self.child.expect(pattern, *args, **kwargs)

    def darwin_log_available(self):
        self.runCmd("plugin structured-data darwin-log status")
        self.expect(re.compile(r"Availability: ([\S]+)"))
        return self.child.match is not None and (
            self.child.match.group(1) == "available")

    def do_test(self, enable_options, expect_regexes=None,
                settings_commands=None):
        """Test that a single fall-through reject rule rejects all logging."""
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        # Build the darwin-log enable command.
        enable_cmd = expand_darwinlog_command('enable')
        if enable_options is not None and len(enable_options) > 0:
            enable_cmd += ' ' + ' '.join(enable_options)

        exe = os.path.join(os.getcwd(), self.exe_name)
        self.run_lldb_to_breakpoint(exe, self.source, self.line,
                                    enable_command=enable_cmd,
                                    settings_commands=settings_commands)
        self.expect_prompt()

        # Now go.
        self.runCmd("process continue")
        self.expect(self.CONTINUE_REGEX)

        if expect_regexes is None:
            # Expect matching a log line or program exit.
            # Test methods determine which ones are valid.
            expect_regexes = (
                [re.compile(r"source-log-([^-]+)-(\S+)"),
                 re.compile(r"exited with status")
                ])
        self.expect(expect_regexes)


def remove_add_mode_entry(log_entries):
    """libtrace creates an "Add Mode:..." message when logging is enabled.
    Strip this out of results since our test subjects don't create it."""
    return [entry for entry in log_entries
            if "message" in entry and
            not entry["message"].startswith("Add Mode:")]


class DarwinLogEventBasedTestBase(lldbtest.TestBase):
    """Base class for event-based DarwinLog tests."""
    NO_DEBUG_INFO_TESTCASE = True

    class EventListenerThread(threading.Thread):
        def __init__(self, listener, process, trace_on, max_entry_count):
            super(DarwinLogEventBasedTestBase.EventListenerThread, self).__init__()
            self.process = process
            self.listener = listener
            self.trace_on = trace_on
            self.max_entry_count = max_entry_count
            self.exception = None
            self.structured_data_event_count = 0
            self.wait_seconds = 2
            self.max_timeout_count = 4
            self.log_entries = []

        def handle_structured_data_event(self, event):
            structured_data = SBProcess.GetStructuredDataFromEvent(event)
            if not structured_data.IsValid():
                if self.trace_on:
                    print("invalid structured data")
                return

            # Track that we received a valid structured data event.
            self.structured_data_event_count += 1

            # Grab the individual log entries from the JSON.
            stream = lldb.SBStream()
            structured_data.GetAsJSON(stream)
            dict = json.loads(stream.GetData())
            self.log_entries.extend(dict["events"])
            if self.trace_on:
                print("Structured data (raw):", stream.GetData())

            # Print the pretty-printed version.
            if self.trace_on:
                stream.Clear()
                structured_data.PrettyPrint(stream)
                print("Structured data (pretty print):",
                      stream.GetData())

        def done(self, timeout_count):
            """Returns True when we're done listening for events."""
            # See if we should consider the number of events retrieved.
            if self.max_entry_count is not None:
                if len(self.log_entries) >= self.max_entry_count:
                    # We've received the max threshold of events expected,
                    # we can exit here.
                    if self.trace_on:
                        print("Event listener thread exiting due to max "
                              "expected log entry count being reached.")
                    return True

            # If our event timeout count has exceeded our maximum timeout count,
            # we're done.
            if timeout_count >= self.max_timeout_count:
                if self.trace_on:
                    print("Event listener thread exiting due to max number of "
                          "WaitForEvent() timeouts being reached.")
                return True

            # If our process is dead, we're done.
            if not self.process.is_alive:
                if self.trace_on:
                    print("Event listener thread exiting due to test inferior "
                          "exiting.")
                return True

            # We're not done.
            return False

        def run(self):
            event = lldb.SBEvent()
            try:
                timeout_count = 0

                # Wait up to 4 times for the event to arrive.
                while not self.done(timeout_count):
                    if self.trace_on:
                        print("Calling wait for event...")
                    if self.listener.WaitForEvent(self.wait_seconds, event):
                        while event.IsValid():
                            # Check if it's a process event.
                            if SBProcess.EventIsStructuredDataEvent(event):
                                self.handle_structured_data_event(event)
                            else:
                                if self.trace_on:
                                    print("ignoring unexpected event:",
                                          lldbutil.get_description(event))
                            # Grab the next event, if there is one.
                            event.Clear()
                            if not self.listener.GetNextEvent(event):
                                if self.trace_on:
                                    print("listener has no more events "
                                          "available at this time")
                    else:
                        if self.trace_on:
                            print("timeout occurred waiting for event...")
                        timeout_count += 1
                self.listener.Clear()
            except Exception as e:
                self.exception = e

    def setUp(self):
        # Call super's setUp().
        super(DarwinLogEventBasedTestBase, self).setUp()

        # Source filename.
        self.source = 'main.c'

        # Output filename.
        self.exe_name = 'a.out'
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

        # Locate breakpoint.
        self.line = lldbtest.line_number(self.source, '// break here')

        # Enable debugserver logging of the darwin log collection
        # mechanism.
        self.runCmd("settings set target.process.extra-startup-command "
                    "QSetLogging:bitmask=LOG_DARWIN_LOG;")

    def do_test(self, enable_options, settings_commands=None,
                run_enable_after_breakpoint=False, max_entry_count=None):
        """Runs the test inferior, returning collected events.

        This method runs the test inferior to the first breakpoint hit.
        It then adds a listener for structured data events, and collects
        all events from that point forward until end of execution of the
        test inferior.  It then returns those events.

        @return
            A list of structured data events received, in the order they
            were received.
        """
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        exe = os.path.join(os.getcwd(), self.exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        # Run the darwin-log settings commands.
        if settings_commands is not None:
            for setting_command in settings_commands:
                self.runCmd(
                    expand_darwinlog_settings_set_command(setting_command))

        # Build the darwin-log enable command.
        enable_cmd = expand_darwinlog_command("enable")
        if enable_options is not None and len(enable_options) > 0:
            enable_cmd += ' ' + ' '.join(enable_options)

        # Run the darwin-log enable command now if we are not supposed
        # to do it at the first breakpoint.  This tests the start-up
        # code, which has the benefit of being able to set os_log-related
        # environment variables.
        if not run_enable_after_breakpoint:
            self.runCmd(enable_cmd)

        # Create the breakpoint.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertIsNotNone(breakpoint)
        self.assertTrue(breakpoint.IsValid())
        self.assertEqual(1, breakpoint.GetNumLocations(),
                         "Should have found one breakpoint")

        # Enable packet logging.
        # self.runCmd("log enable -f /tmp/packets.log gdb-remote packets")
        # self.runCmd("log enable lldb process")

        # Launch the process - doesn't stop at entry.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertIsNotNone(process, lldbtest.PROCESS_IS_VALID)

        # Keep track of whether we're tracing output.
        trace_on = self.TraceOn()

        # Get the next thread that stops.
        from lldbsuite.test.lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        self.assertIsNotNone(thread, "There should be a thread stopped "
                             "due to breakpoint")

        # The process should be stopped at this point.
        self.expect("process status", lldbtest.PROCESS_STOPPED,
                    patterns=['Process .* stopped'])

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", lldbtest.STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        # And our one and only breakpoint should have been hit.
        self.assertEquals(breakpoint.GetHitCount(), 1)

        # Now setup the structured data listener.
        #
        # Grab the broadcaster for the process.  We'll be attaching our
        # listener to it.
        broadcaster = process.GetBroadcaster()
        self.assertIsNotNone(broadcaster)

        listener = lldb.SBListener("SBStructuredData listener")
        self.assertIsNotNone(listener)

        rc = broadcaster.AddListener(listener,
                                     lldb.SBProcess.eBroadcastBitStructuredData)
        self.assertTrue(rc, "Successfully add listener to process broadcaster")

        # Start the listening thread to retrieve the events.
        # Bump up max entry count for the potentially included Add Mode:
        # entry.
        if max_entry_count is not None:
            max_entry_count += 1
        event_thread = self.EventListenerThread(listener, process, trace_on,
                                                max_entry_count)
        event_thread.start()

        # Continue the test inferior.  We should get any events after this.
        process.Continue()

        # Wait until the event thread terminates.
        # print("main thread now waiting for event thread to receive events.")
        event_thread.join()

        # If the process is still alive, we kill it here.
        if process.is_alive:
            process.Kill()

        # Fail on any exceptions that occurred during event execution.
        if event_thread.exception is not None:
            # Re-raise it here so it shows up as a test error.
            raise event_thread

        # Return the collected logging events.
        return remove_add_mode_entry(event_thread.log_entries)
