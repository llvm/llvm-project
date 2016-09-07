"""
Interprocess mutex based on file locks
"""

import fcntl
import os


class Lock:

    def __init__(self, filename):
        self.filename = filename
        # This will create it if it does not exist already
        unbuffered = 0
        self.handle = open(filename, 'a+', unbuffered)

    def acquire(self):
        fcntl.flock(self.handle, fcntl.LOCK_EX)

    # will throw IOError if unavailable
    def try_acquire(self):
        fcntl.flock(self.handle, fcntl.LOCK_NB | fcntl.LOCK_EX)

    def release(self):
        fcntl.flock(self.handle, fcntl.LOCK_UN)

    def __del__(self):
        self.handle.close()
