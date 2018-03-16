import lldb
import struct


class OperatingSystemPlugIn(object):
    """Class that provides data for an instance of a LLDB 'OperatingSystemPython' plug-in class"""

    def __init__(self, process):
        '''Initialization needs a valid.SBProcess object.

        This plug-in will get created after a live process is valid and has stopped for the first time.
        '''
        self.process = None
        self.registers = None
        self.threads = None
        if isinstance(process, lldb.SBProcess) and process.IsValid():
            self.process = process
            self.threads = None  # Will be an dictionary containing info for each thread

    def get_target(self):
        return self.process.target

    def get_thread_info(self):
        if not self.threads:
            self.threads = [{
                'tid': 0x111111111,
                'name': 'one',
                'queue': 'queue1',
                'state': 'stopped',
                'stop_reason': 'none'
            }, {
                'tid': 0x222222222,
                'name': 'two',
                'queue': 'queue2',
                'state': 'stopped',
                'stop_reason': 'none'
            }, {
                'tid': 0x333333333,
                'name': 'three',
                'queue': 'queue3',
                'state': 'stopped',
                'stop_reason': 'sigstop',
                'core': 0
            }]
        return self.threads
