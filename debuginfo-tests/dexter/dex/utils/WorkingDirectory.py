# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Create/set a temporary working directory for some operations."""

import os
import shutil
import tempfile
import time

from dex.utils.Exceptions import Error


class WorkingDirectory(object):
    def __init__(self, context, *args, **kwargs):
        self.context = context
        self.orig_cwd = os.getcwd()

        dir_ = kwargs.get('dir', None)
        if dir_ and not os.path.isdir(dir_):
            os.makedirs(dir_, exist_ok=True)
        self.path = tempfile.mkdtemp(*args, **kwargs)

    def __enter__(self):
        os.chdir(self.path)
        return self

    def __exit__(self, *args):
        os.chdir(self.orig_cwd)
        if self.context.options.save_temps:
            self.context.o.blue('"{}" left in place [--save-temps]\n'.format(
                self.path))
            return

        exception = AssertionError('should never be raised')
        for _ in range(100):
            try:
                shutil.rmtree(self.path)
                return
            except OSError as e:
                exception = e
                time.sleep(0.1)
        raise Error(exception)
