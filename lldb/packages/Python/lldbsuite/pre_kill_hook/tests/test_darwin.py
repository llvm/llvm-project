"""Test the pre-kill hook on Darwin."""
from __future__ import print_function

# system imports
from multiprocessing import Process, Queue
import platform
import re
from unittest import main, TestCase

# third party
from six import StringIO


def do_child_process(child_work_queue, parent_work_queue, verbose):
    import os

    pid = os.getpid()
    if verbose:
        print("child: pid {} started, sending to parent".format(pid))
    parent_work_queue.put(pid)
    if verbose:
        print("child: waiting for shut-down request from parent")
    child_work_queue.get()
    if verbose:
        print("child: received shut-down request.  Child exiting.")


class DarwinPreKillTestCase(TestCase):

    def __init__(self, methodName):
        super(DarwinPreKillTestCase, self).__init__(methodName)
        self.process = None
        self.child_work_queue = None
        self.verbose = False

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
        if platform.system() != 'Darwin':
            self.skipTest("requires a Darwin-based OS")

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
        from darwin import do_pre_kill
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

        # We should have a line with:
        # Process:  .* [{pid}]
        process_re = re.compile(r"Process:[^[]+\[([^]]+)\]")
        match = process_re.search(output)
        self.assertIsNotNone(match, "should have found process id for "
                             "sampled process")
        self.assertEqual(1, len(match.groups()))
        self.assertEqual(child_pid, int(match.group(1)))

        # We should see a Call graph: section.
        callgraph_re = re.compile(r"Call graph:")
        match = callgraph_re.search(output)
        self.assertIsNotNone(match, "should have found the Call graph section"
                             "in sample output")

        # We should see a Binary Images: section.
        binary_images_re = re.compile(r"Binary Images:")
        match = binary_images_re.search(output)
        self.assertIsNotNone(match, "should have found the Binary Images "
                             "section in sample output")


if __name__ == "__main__":
    main()
