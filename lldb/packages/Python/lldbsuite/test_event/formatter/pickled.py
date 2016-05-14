"""
    The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.
"""

from __future__ import print_function
from __future__ import absolute_import

# System modules
import os

# Our modules
from .results_formatter import ResultsFormatter
from six.moves import cPickle


class RawPickledFormatter(ResultsFormatter):
    """Formats events as a pickled stream.

    The parallel test runner has inferiors pickle their results and send them
    over a socket back to the parallel test.  The parallel test runner then
    aggregates them into the final results formatter (e.g. xUnit).
    """

    @classmethod
    def arg_parser(cls):
        """@return arg parser used to parse formatter-specific options."""
        parser = super(RawPickledFormatter, cls).arg_parser()
        return parser

    def __init__(self, out_file, options):
        super(RawPickledFormatter, self).__init__(out_file, options)
        self.pid = os.getpid()

    def handle_event(self, test_event):
        super(RawPickledFormatter, self).handle_event(test_event)

        # Convert initialize/terminate events into job_begin/job_end events.
        event_type = test_event["event"]
        if event_type is None:
            return

        if event_type == "initialize":
            test_event["event"] = "job_begin"
        elif event_type == "terminate":
            test_event["event"] = "job_end"

        # Tack on the pid.
        test_event["pid"] = self.pid

        # Send it as {serialized_length_of_serialized_bytes}{serialized_bytes}
        import struct
        msg = cPickle.dumps(test_event)
        packet = struct.pack("!I%ds" % len(msg), len(msg), msg)
        self.out_file.send(packet)
