"""
                     The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.

Sync lldb and related source from a local machine to a remote machine.

This facilitates working on the lldb sourcecode on multiple machines
and multiple OS types, verifying changes across all.


This module provides asyncore channels used within the LLDB test
framework.
"""

from __future__ import print_function
from __future__ import absolute_import


# System modules
import asyncore
import socket

# Third-party modules
from six.moves import cPickle

# LLDB modules


class UnpicklingForwardingReaderChannel(asyncore.dispatcher):
    """Provides an unpickling, forwarding asyncore dispatch channel reader.

    Inferior dotest.py processes with side-channel-based test results will
    send test result event data in a pickled format, one event at a time.
    This class supports reconstructing the pickled data and forwarding it
    on to its final destination.

    The channel data is written in the form:
    {num_payload_bytes}#{payload_bytes}

    The bulk of this class is devoted to reading and parsing out
    the payload bytes.
    """
    def __init__(self, file_object, async_map, forwarding_func):
        asyncore.dispatcher.__init__(self, sock=file_object, map=async_map)

        self.header_contents = b""
        self.packet_bytes_remaining = 0
        self.reading_header = True
        self.ibuffer = b''
        self.forwarding_func = forwarding_func
        if forwarding_func is None:
            # This whole class is useless if we do nothing with the
            # unpickled results.
            raise Exception("forwarding function must be set")

    def deserialize_payload(self):
        """Unpickles the collected input buffer bytes and forwards."""
        if len(self.ibuffer) > 0:
            self.forwarding_func(cPickle.loads(self.ibuffer))
            self.ibuffer = b''

    def consume_header_bytes(self, data):
        """Consumes header bytes from the front of data.
        @param data the incoming data stream bytes
        @return any data leftover after consuming header bytes.
        """
        # We're done if there is no content.
        if not data or (len(data) == 0):
            return None

        full_header_len = 4

        assert len(self.header_contents) < full_header_len

        bytes_avail = len(data)
        bytes_needed = full_header_len - len(self.header_contents)
        header_bytes_avail = min(bytes_needed, bytes_avail)
        self.header_contents += data[:header_bytes_avail]
        if len(self.header_contents) == full_header_len:
            import struct
            # End of header.
            self.packet_bytes_remaining = struct.unpack(
                "!I", self.header_contents)[0]
            self.header_contents = b""
            self.reading_header = False
            return data[header_bytes_avail:]

        # If we made it here, we've exhausted the data and
        # we're still parsing header content.
        return None

    def consume_payload_bytes(self, data):
        """Consumes payload bytes from the front of data.
        @param data the incoming data stream bytes
        @return any data leftover after consuming remaining payload bytes.
        """
        if not data or (len(data) == 0):
            # We're done and there's nothing to do.
            return None

        data_len = len(data)
        if data_len <= self.packet_bytes_remaining:
            # We're consuming all the data provided.
            self.ibuffer += data
            self.packet_bytes_remaining -= data_len

            # If we're no longer waiting for payload bytes,
            # we flip back to parsing header bytes and we
            # unpickle the payload contents.
            if self.packet_bytes_remaining < 1:
                self.reading_header = True
                self.deserialize_payload()

            # We're done, no more data left.
            return None
        else:
            # We're only consuming a portion of the data since
            # the data contains more than the payload amount.
            self.ibuffer += data[:self.packet_bytes_remaining]
            data = data[self.packet_bytes_remaining:]

            # We now move on to reading the header.
            self.reading_header = True
            self.packet_bytes_remaining = 0

            # And we can deserialize the payload.
            self.deserialize_payload()

            # Return the remaining data.
            return data

    def handle_read(self):
        # Read some data from the socket.
        try:
            data = self.recv(8192)
            # print('driver socket READ: %d bytes' % len(data))
        except socket.error as socket_error:
            print(
                "\nINFO: received socket error when reading data "
                "from test inferior:\n{}".format(socket_error))
            raise
        except Exception as general_exception:
            print(
                "\nERROR: received non-socket error when reading data "
                "from the test inferior:\n{}".format(general_exception))
            raise

        # Consume the message content.
        while data and (len(data) > 0):
            # If we're reading the header, gather header bytes.
            if self.reading_header:
                data = self.consume_header_bytes(data)
            else:
                data = self.consume_payload_bytes(data)

    def handle_close(self):
        # print("socket reader: closing port")
        self.close()


class UnpicklingForwardingListenerChannel(asyncore.dispatcher):
    """Provides a socket listener asyncore channel for unpickling/forwarding.

    This channel will listen on a socket port (use 0 for host-selected).  Any
    client that connects will have an UnpicklingForwardingReaderChannel handle
    communication over the connection.

    The dotest parallel test runners, when collecting test results, open the
    test results side channel over a socket.  This channel handles connections
    from inferiors back to the test runner.  Each worker fires up a listener
    for each inferior invocation.  This simplifies the asyncore.loop() usage,
    one of the reasons for implementing with asyncore.  This listener shuts
    down once a single connection is made to it.
    """
    def __init__(self, async_map, host, port, backlog_count, forwarding_func):
        asyncore.dispatcher.__init__(self, map=async_map)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind((host, port))
        self.address = self.socket.getsockname()
        self.listen(backlog_count)
        self.handler = None
        self.async_map = async_map
        self.forwarding_func = forwarding_func
        if forwarding_func is None:
            # This whole class is useless if we do nothing with the
            # unpickled results.
            raise Exception("forwarding function must be set")

    def handle_accept(self):
        (sock, addr) = self.socket.accept()
        if sock and addr:
            # print('Incoming connection from %s' % repr(addr))
            self.handler = UnpicklingForwardingReaderChannel(
                sock, self.async_map, self.forwarding_func)

    def handle_close(self):
        self.close()
