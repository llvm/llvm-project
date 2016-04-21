"""
    The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.
"""

from __future__ import print_function
from __future__ import absolute_import

# System modules
import pprint

# Our modules
from .results_formatter import ResultsFormatter


class DumpFormatter(ResultsFormatter):
    """Formats events to the file as their raw python dictionary format."""

    def handle_event(self, test_event):
        super(DumpFormatter, self).handle_event(test_event)
        self.out_file.write("\n" + pprint.pformat(test_event) + "\n")
