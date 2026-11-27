# System modules
import os
import sys
import time
import signal

# LLDB Modules
import lldb
from .lldbtest import *
from . import lldbutil
from lldbsuite.test.decorators import *


@skipIfRemote
@skipIfWindows
@add_test_categories(["pexpect"])
class PExpectTest(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    PROMPT = "(lldb) "

    # Override this value in a subclass to make the test fail faster and make
    # debugging less tedious.
    TIMEOUT = 60

    def expect_prompt(self):
        self.child.expect_exact(self.PROMPT)

    def launch(
        self,
        executable=None,
        extra_args=None,
        dimensions=None,
        run_under=None,
        post_spawn=None,
        encoding=None,
        use_colors=False,
    ):
        # Using a log file is incompatible with using utf-8 as the encoding.
        logfile = (
            getattr(sys.stdout, "buffer", sys.stdout)
            if (self.TraceOn() and not encoding)
            else None
        )

        args = []
        if run_under is not None:
            args += run_under
        args += [lldbtest_config.lldbExec, "--no-lldbinit"]
        if not use_colors:
            args.append("--no-use-colors")
        for cmd in self.setUpCommands():
            if "use-color false" in cmd and use_colors:
                continue
            args += ["-O", cmd]
        if executable is not None:
            args += ["--file", executable]
        if extra_args is not None:
            args.extend(extra_args)

        env = dict(os.environ)
        env["TERM"] = "vt100"
        env["HOME"] = self.getBuildDir()

        timeout = self.TIMEOUT

        import pexpect

        self.child = pexpect.spawn(
            args[0],
            args=args[1:],
            logfile=logfile,
            timeout=timeout,
            dimensions=dimensions,
            env=env,
            encoding=encoding,
        )

        # The interval in which we poll the LLDB process if we are
        # closing/terminating the process.
        self.polling_interval = 0.05
        # Timeout for each attempt to shut down LLDB at the end
        # of the test. Note that there can be multiple attempts
        # to shut down LLDB.
        self.shutdown_timeout = timeout / 10

        if post_spawn is not None:
            post_spawn()
        self.expect_prompt()
        for cmd in self.setUpCommands():
            if "use-color false" in cmd and use_colors:
                continue
            self.child.expect_exact(cmd)
            self.expect_prompt()
        if executable is not None:
            self.child.expect_exact("target create")
            self.child.expect_exact("Current executable set to")
            self.expect_prompt()

    def expect(self, cmd, substrs=None):
        self.assertNotIn("\n", cmd)
        # If 'substrs' is a string then this code would just check that every
        # character of the string is in the output.
        assert not isinstance(substrs, str), "substrs must be a collection of strings"

        self.child.sendline(cmd)
        if substrs is not None:
            for s in substrs:
                self.child.expect_exact(s)
        self.expect_prompt()

    def tearDown(self):
        # Ensure the child is always cleaned up, even if the test didn't call
        # quit() explicitly or failed before reaching it.
        if self.child is not None:
            self.quit(gracefully=True)
        super().tearDown()

    def _poll_until_dead(self, proc):
        """Poll proc.isalive() for up to shutdown_timeout seconds.
        Returns True if the process exited within that window."""
        deadline = time.monotonic() + self.shutdown_timeout
        while time.monotonic() < deadline:
            if not proc.isalive():
                return True
            time.sleep(self.polling_interval)
        return not proc.isalive()

    def _terminate_child(self, force=False):
        """Terminate the child using signals."""
        proc = self.child.ptyproc

        if not proc.isalive():
            return True
        try:
            # List of signals to try, starting from a friendly
            # SIGHUP, SIGINT to SIGKILL if all else fails.
            signals = [signal.SIGHUP, signal.SIGINT]
            if force:
                signals.append(signal.SIGKILL)
            for sig in signals:
                proc.kill(sig)
                if self._poll_until_dead(proc):
                    return True
            return False
        except OSError:
            # This handles a TOCTOU issue where we think the process is
            # considered as 'isalive', then is actually killed by the
            # kernel and then we try to kill it.
            time.sleep(self.polling_interval)
            return not proc.isalive()

    def _close_child(self, force=False):
        """Close the connection to the child and terminate it if needed."""

        ptyproc = self.child.ptyproc
        if ptyproc.closed:
            return

        # Close the PTY master fd first.
        try:
            self.child.sendeof()
            ptyproc.fileobj.close()
        except OSError:
            # This could happen if LLDB shut down at the same time.
            pass

        if not force:
            # Poll briefly for the process to respond to the EOF before
            # escalating to signals.
            self._poll_until_dead(ptyproc)

        if ptyproc.isalive():
            # Kill the child with signals. This should always succeed.
            assert self._terminate_child(force=force)

        # Make sure the object is now marked as closed.
        ptyproc.fd = -1
        ptyproc.closed = True

    def quit(self, gracefully=True):
        if self.child is None:
            return

        try:
            self._close_child(force=not gracefully)
        except OSError:
            # This can happen if LLDB quit itself because it is running in
            # batch mode.
            pass
        self.child = None

    def cursor_forward_escape_seq(self, chars_to_move):
        """
        Returns the escape sequence to move the cursor forward/right
        by a certain amount of characters.
        """
        return b"\x1b\\[" + str(chars_to_move).encode("utf-8") + b"C"
