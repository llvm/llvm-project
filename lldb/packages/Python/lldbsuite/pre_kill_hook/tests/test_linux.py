"""Test the pre-kill hook on Linux."""
from __future__ import print_function

# system imports
from multiprocessing import Process, Queue
import platform
import re
import subprocess
from unittest import main, TestCase

# third party
from six import StringIO


def do_child_thread():
    import os
    x = 0
    while True:
        x = x + 42 * os.getpid()
    return x


def do_child_process(child_work_queue, parent_work_queue, verbose):
    import os

    pid = os.getpid()
    if verbose:
        print("child: pid {} started, sending to parent".format(pid))
    parent_work_queue.put(pid)

    # Spin up a daemon thread to do some "work", which will show
    # up in a sample of this process.
    import threading
    worker = threading.Thread(target=do_child_thread)
    worker.daemon = True
    worker.start()

    if verbose:
        print("child: waiting for shut-down request from parent")
    child_work_queue.get()
    if verbose:
        print("child: received shut-down request.  Child exiting.")


class LinuxPreKillTestCase(TestCase):

    def __init__(self, methodName):
        super(LinuxPreKillTestCase, self).__init__(methodName)
        self.process = None
        self.child_work_queue = None
        self.verbose = False
        # self.verbose = True

    def tearDown(self):
        if self.verbose:
            print("parent: sending shut-down request to child")
        if self.process:
            self.child_work_queue.put("hello, child")
            self.process.join()
        if self.verbose:
            print("parent: child is fully shut down")

    def test_sample(self):
        # Ensure we're Darwin.
        if platform.system() != 'Linux':
            self.skipTest("requires a Linux-based OS")

        # Ensure we have the 'perf' tool.  If not, skip the test.
        try:
            perf_version = subprocess.check_output(["perf", "version"])
            if perf_version is None or not (
                    perf_version.startswith("perf version")):
                raise Exception("The perf executable doesn't appear"
                                " to be the Linux perf tools perf")
        except Exception:
            self.skipTest("requires the Linux perf tools 'perf' command")

        # Start the child process.
        self.child_work_queue = Queue()
        parent_work_queue = Queue()
        self.process = Process(target=do_child_process,
                               args=(self.child_work_queue, parent_work_queue,
                                     self.verbose))
        if self.verbose:
            print("parent: starting child")
        self.process.start()

        # Wait for the child to report its pid.  Then we know we're running.
        if self.verbose:
            print("parent: waiting for child to start")
        child_pid = parent_work_queue.get()

        # Sample the child process.
        from linux import do_pre_kill
        context_dict = {
            "archs": [platform.machine()],
            "platform_name": None,
            "platform_url": None,
            "platform_working_dir": None
        }

        if self.verbose:
            print("parent: running pre-kill action on child")
        output_io = StringIO()
        do_pre_kill(child_pid, context_dict, output_io)
        output = output_io.getvalue()

        if self.verbose:
            print("parent: do_pre_kill() wrote the following output:", output)
        self.assertIsNotNone(output)

        # We should have a samples count entry.
        # Samples:
        self.assertTrue("Samples:" in output, "should have found a 'Samples:' "
                        "field in the sampled process output")

        # We should see an event count entry
        event_count_re = re.compile(r"Event count[^:]+:\s+(\d+)")
        match = event_count_re.search(output)
        self.assertIsNotNone(match, "should have found the event count entry "
                             "in sample output")
        if self.verbose:
            print("cpu-clock events:", match.group(1))

        # We should see some percentages in the file.
        percentage_re = re.compile(r"\d+\.\d+%")
        match = percentage_re.search(output)
        self.assertIsNotNone(match, "should have found at least one percentage "
                             "in the sample output")


if __name__ == "__main__":
    main()
