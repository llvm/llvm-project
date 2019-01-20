"""
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
