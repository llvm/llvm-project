#!/usr/bin/env python
"""
The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.

Provides classes used by the test results reporting infrastructure
within the LLDB test suite.


Tests the process_control module.
"""

# System imports.
import os
import platform
import unittest
import sys
import threading

# Add lib dir to pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

# Our imports.
import process_control


class TestInferiorDriver(process_control.ProcessDriver):
    def __init__(self, soft_terminate_timeout=None):
        super(TestInferiorDriver, self).__init__(
            soft_terminate_timeout=soft_terminate_timeout)
        self.started_event = threading.Event()
        self.started_event.clear()

        self.completed_event = threading.Event()
        self.completed_event.clear()

        self.was_timeout = False
        self.returncode = None
        self.output = None

    def write(self, content):
        # We'll swallow this to keep tests non-noisy.
        # Uncomment the following line if you want to see it.
        # sys.stdout.write(content)
        pass

    def on_process_started(self):
        self.started_event.set()

    def on_process_exited(self, command, output, was_timeout, exit_status):
        self.returncode = exit_status
        self.was_timeout = was_timeout
        self.output = output
        self.returncode = exit_status
        self.completed_event.set()


class ProcessControlTests(unittest.TestCase):
    @classmethod
    def _suppress_soft_terminate(cls, command):
        # Do the right thing for your platform here.
        # Right now only POSIX-y systems are reporting
        # soft terminate support, so this is set up for
        # those.
        helper = process_control.ProcessHelper.process_helper()
        signals = helper.soft_terminate_signals()
        if signals is not None:
            for signum in helper.soft_terminate_signals():
                command.extend(["--ignore-signal", str(signum)])

    @classmethod
    def inferior_command(
            cls,
            ignore_soft_terminate=False,
            options=None):

        # Base command.
        command = ([sys.executable, "inferior.py"])

        if ignore_soft_terminate:
            cls._suppress_soft_terminate(command)

        # Handle options as string or list.
        if isinstance(options, str):
            command.extend(options.split())
        elif isinstance(options, list):
            command.extend(options)

        # Return full command.
        return command


class ProcessControlNoTimeoutTests(ProcessControlTests):
    """Tests the process_control module."""
    def test_run_completes(self):
        """Test that running completes and gets expected stdout/stderr."""
        driver = TestInferiorDriver()
        driver.run_command(self.inferior_command())
        self.assertTrue(
            driver.completed_event.wait(5), "process failed to complete")
        self.assertEqual(driver.returncode, 0, "return code does not match")

    def test_run_completes_with_code(self):
        """Test that running completes and gets expected stdout/stderr."""
        driver = TestInferiorDriver()
        driver.run_command(self.inferior_command(options="-r10"))
        self.assertTrue(
            driver.completed_event.wait(5), "process failed to complete")
        self.assertEqual(driver.returncode, 10, "return code does not match")


class ProcessControlTimeoutTests(ProcessControlTests):
    def test_run_completes(self):
        """Test that running completes and gets expected return code."""
        driver = TestInferiorDriver()
        timeout_seconds = 5
        driver.run_command_with_timeout(
            self.inferior_command(),
            "{}s".format(timeout_seconds),
            False)
        self.assertTrue(
            driver.completed_event.wait(2*timeout_seconds),
            "process failed to complete")
        self.assertEqual(driver.returncode, 0)

    def _soft_terminate_works(self, with_core):
        # Skip this test if the platform doesn't support soft ti
        helper = process_control.ProcessHelper.process_helper()
        if not helper.supports_soft_terminate():
            self.skipTest("soft terminate not supported by platform")

        driver = TestInferiorDriver()
        timeout_seconds = 5

        driver.run_command_with_timeout(
            # Sleep twice as long as the timeout interval.  This
            # should force a timeout.
            self.inferior_command(
                options="--sleep {}".format(timeout_seconds*2)),
            "{}s".format(timeout_seconds),
            with_core)

        # We should complete, albeit with a timeout.
        self.assertTrue(
            driver.completed_event.wait(2*timeout_seconds),
            "process failed to complete")

        # Ensure we received a timeout.
        self.assertTrue(driver.was_timeout, "expected to end with a timeout")

        self.assertTrue(
            helper.was_soft_terminate(driver.returncode, with_core),
            ("timeout didn't return expected returncode "
             "for soft terminate with core: {}").format(driver.returncode))

    def test_soft_terminate_works_core(self):
        """Driver uses soft terminate (with core request) when process times out.
        """
        self._soft_terminate_works(True)

    def test_soft_terminate_works_no_core(self):
        """Driver uses soft terminate (no core request) when process times out.
        """
        self._soft_terminate_works(False)

    def test_hard_terminate_works(self):
        """Driver falls back to hard terminate when soft terminate is ignored.
        """

        driver = TestInferiorDriver(soft_terminate_timeout=2.0)
        timeout_seconds = 1

        driver.run_command_with_timeout(
            # Sleep much longer than the timeout interval,forcing a
            # timeout.  Do whatever is needed to have the inferior
            # ignore soft terminate calls.
            self.inferior_command(
                ignore_soft_terminate=True,
                options="--never-return"),
            "{}s".format(timeout_seconds),
            True)

        # We should complete, albeit with a timeout.
        self.assertTrue(
            driver.completed_event.wait(60),
            "process failed to complete")

        # Ensure we received a timeout.
        self.assertTrue(driver.was_timeout, "expected to end with a timeout")

        helper = process_control.ProcessHelper.process_helper()
        self.assertTrue(
            helper.was_hard_terminate(driver.returncode),
            ("timeout didn't return expected returncode "
             "for hard teriminate: {} ({})").format(
                 driver.returncode,
                 driver.output))

    def test_inferior_exits_with_live_child_shared_handles(self):
        """inferior exit detected when inferior children are live with shared
        stdout/stderr handles.
        """
        # Requires review D13362 or equivalent to be implemented.
        self.skipTest("http://reviews.llvm.org/D13362")

        driver = TestInferiorDriver()

        # Create the inferior (I1), and instruct it to create a child (C1)
        # that shares the stdout/stderr handles with the inferior.
        # C1 will then loop forever.
        driver.run_command_with_timeout(
            self.inferior_command(
                options="--launch-child-share-handles --return-code 3"),
            "5s",
            False)

        # We should complete without a timetout.  I1 should end
        # immediately after launching C1.
        self.assertTrue(
            driver.completed_event.wait(5),
            "process failed to complete")

        # Ensure we didn't receive a timeout.
        self.assertFalse(
            driver.was_timeout, "inferior should have completed normally")

        self.assertEqual(
            driver.returncode, 3,
            "expected inferior process to end with expected returncode")


if __name__ == "__main__":
    unittest.main()
