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
class FormatterConfig(object):
    """Provides formatter configuration info to create_results_formatter()."""

    def __init__(self):
        self.filename = None
        self.port = None
        self.formatter_name = None
        self.formatter_options = None


# Ignore method count on DTOs.
# pylint: disable=too-few-public-methods
class CreatedFormatter(object):
    """Provides transfer object for returns from create_results_formatter()."""

    def __init__(self, formatter, cleanup_func):
        self.formatter = formatter
        self.cleanup_func = cleanup_func


SOCKET_ACK_BYTE_VALUE = b'*'  # ASCII for chr(42)


def create_results_formatter(config):
    """Sets up a test results formatter.

    @param config an instance of FormatterConfig
    that indicates how to setup the ResultsFormatter.

    @return an instance of CreatedFormatter.
    """

    def create_socket(port):
        """Creates a socket to the localhost on the given port.

        @param port the port number of the listening port on
        the localhost.

        @return (socket object, socket closing function)
        """

        def socket_closer(open_sock):
            """Close down an opened socket properly."""
            open_sock.shutdown(socket.SHUT_RDWR)
            open_sock.close()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("localhost", port))

        # Wait for the ack from the listener side.
        # This is needed to prevent a race condition
        # in the main dosep.py processing loop: we
        # can't allow a worker queue thread to die
        # that has outstanding messages to a listener
        # socket before the listener socket asyncore
        # listener socket gets spun up; otherwise,
        # we lose the test result info.
        read_bytes = sock.recv(1)
        if read_bytes is None or (
                len(read_bytes) < 1) or (
                read_bytes[0] != SOCKET_ACK_BYTE_VALUE):
            raise Exception(
                "listening socket did not respond with ack byte: response={}".format(read_bytes))

        return sock, lambda: socket_closer(sock)

    default_formatter_name = None
    results_file_object = None
    cleanup_func = None

    file_is_stream = False
    if config.filename:
        # Open the results file for writing.
        if config.filename == 'stdout':
            results_file_object = sys.stdout
            cleanup_func = None
        elif config.filename == 'stderr':
            results_file_object = sys.stderr
            cleanup_func = None
        else:
            results_file_object = open(config.filename, "w")
            cleanup_func = results_file_object.close
        default_formatter_name = (
            "lldbsuite.test_event.formatter.xunit.XunitFormatter")
    elif config.port:
        # Connect to the specified localhost port.
        results_file_object, cleanup_func = create_socket(config.port)
        default_formatter_name = (
            "lldbsuite.test_event.formatter.pickled.RawPickledFormatter")
        file_is_stream = True

    # If we have a results formatter name specified and we didn't specify
    # a results file, we should use stdout.
    if config.formatter_name is not None and results_file_object is None:
        # Use stdout.
        results_file_object = sys.stdout
        cleanup_func = None

    if results_file_object:
        # We care about the formatter.  Choose user-specified or, if
        # none specified, use the default for the output type.
        if config.formatter_name:
            formatter_name = config.formatter_name
        else:
            formatter_name = default_formatter_name

        # Create an instance of the class.
        # First figure out the package/module.
        components = formatter_name.split(".")
        module = importlib.import_module(".".join(components[:-1]))

        # Create the class name we need to load.
        cls = getattr(module, components[-1])

        # Handle formatter options for the results formatter class.
        formatter_arg_parser = cls.arg_parser()
        if config.formatter_options and len(config.formatter_options) > 0:
            command_line_options = config.formatter_options
        else:
            command_line_options = []

        formatter_options = formatter_arg_parser.parse_args(
            command_line_options)

        # Create the TestResultsFormatter given the processed options.
        results_formatter_object = cls(
            results_file_object,
            formatter_options,
            file_is_stream)

        def shutdown_formatter():
            """Shuts down the formatter when it is no longer needed."""
            # Tell the formatter to write out anything it may have
            # been saving until the very end (e.g. xUnit results
            # can't complete its output until this point).
            results_formatter_object.send_terminate_as_needed()

            # And now close out the output file-like object.
            if cleanup_func is not None:
                cleanup_func()

        return CreatedFormatter(
            results_formatter_object,
            shutdown_formatter)
    else:
        return None
