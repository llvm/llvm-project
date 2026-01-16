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

    def __del__(self):
        if self.path and os.path.exists(self.path):
            os.remove(self.path)

    def _set_path(self):
        if self.path:
            return
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.path = path

    def __enter__(self):
        self._set_path()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
