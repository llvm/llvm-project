"""
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from __future__ import print_function
from __future__ import absolute_import

# System modules
import importlib
import socket
import sys

# Third-party modules

# LLDB modules


# Ignore method count on DTOs.
# pylint: disable=too-few-public-methods
class CreatedFormatter(object):
    """Provides transfer object for returns from create_results_formatter()."""

    def __init__(self, formatter, cleanup_func):
        self.formatter = formatter
        self.cleanup_func = cleanup_func


def create_results_formatter(formatter_name):
    """Sets up a test results formatter.

    @param config an instance of FormatterConfig
    that indicates how to setup the ResultsFormatter.

    @return an instance of CreatedFormatter.
    """

    # Create an instance of the class.
    # First figure out the package/module.
    components = formatter_name.split(".")
    module = importlib.import_module(".".join(components[:-1]))

    # Create the class name we need to load.
    cls = getattr(module, components[-1])

    # Handle formatter options for the results formatter class.
    formatter_arg_parser = cls.arg_parser()
    command_line_options = []

    formatter_options = formatter_arg_parser.parse_args(
        command_line_options)

    # Create the TestResultsFormatter given the processed options.
    results_formatter_object = cls(sys.stdout, formatter_options)

    def shutdown_formatter():
        """Shuts down the formatter when it is no longer needed."""
        # Tell the formatter to write out anything it may have
        # been saving until the very end (e.g. xUnit results
        # can't complete its output until this point).
        results_formatter_object.send_terminate_as_needed()

    return CreatedFormatter(
        results_formatter_object,
        shutdown_formatter)
