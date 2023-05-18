# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utility class to check for timeouts. Timer starts when the object is initialized,
and can be checked by calling timed_out(). Passing a timeout value of 0.0 or less
means a timeout will never be triggered, i.e. timed_out() will always return False.
"""

import time


class Timeout(object):
    def __init__(self, duration: float):
        self.start = self.now
        self.duration = duration

    def timed_out(self):
        if self.duration <= 0.0:
            return False
        return self.elapsed > self.duration

    @property
    def elapsed(self):
        return self.now - self.start

    @property
    def now(self):
        return time.time()
