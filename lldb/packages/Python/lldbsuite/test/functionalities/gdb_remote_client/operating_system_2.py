import lldb
import struct


class OperatingSystemPlugIn(object):
    """Class that provides data for an instance of a LLDB 'OperatingSystemPython' plug-in class
       This version stops once with threads 0x111 and 0x222, then stops a second time with threads
       0x111 and 0x333."""

    def __init__(self, process):
        '''Initialization needs a valid.SBProcess object.

        This plug-in will get created after a live process is valid and has stopped for the first time.
        '''
        self.process = None
        self.registers = None
        self.threads = None
        self.times_called = 0
        if isinstance(process, lldb.SBProcess) and process.IsValid():
            self.process = process
            self.threads = None  # Will be an dictionary containing info for each thread

    def get_target(self):
        return self.process.target

    def get_thread_info(self):
        self.times_called += 1

        if self.times_called == 1:
            self.threads = [{
                'tid': 0x111,
                'name': 'one',
                'queue': 'queue1',
                'state': 'stopped',
                'stop_reason': 'none',
                'core': 1
            }, {
                'tid': 0x222,
                'name': 'two',
                'queue': 'queue2',
                'state': 'stopped',
                'stop_reason': 'none',
                'core': 0
            }]
        else:
            self.threads = [{
                'tid': 0x111,
                'name': 'one',
                'queue': 'queue1',
                'state': 'stopped',
                'stop_reason': 'none',
                'core': 1
            }, {
                'tid': 0x333,
                'name': 'three',
                'queue': 'queue3',
                'state': 'stopped',
                'stop_reason': 'none',
                'core': 0
            }]
        return self.threads

