"""
                     The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.

Helper functions for working with sockets.
"""

# Python modules:
import io
import socket

# LLDB modules
import use_lldb_suite


def recvall(sock, size):
    bytes = io.BytesIO()
    while size > 0:
        this_result = sock.recv(size)
        bytes.write(this_result)
        size -= len(this_result)
    return bytes.getvalue()
