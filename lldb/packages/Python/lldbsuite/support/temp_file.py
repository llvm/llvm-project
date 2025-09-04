"""
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import tempfile


class OnDiskTempFile:
    def __init__(self, delete=True):
        self.path = None

    def __enter__(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.path = path
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.path):
            os.remove(self.path)
