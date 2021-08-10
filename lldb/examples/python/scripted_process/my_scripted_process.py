import os

import lldb
from lldb.plugins.scripted_process import ScriptedProcess

class MyScriptedProcess(ScriptedProcess):
    def __init__(self, target: lldb.SBTarget, args : lldb.SBStructuredData):
        super().__init__(target, args)

    def get_memory_region_containing_address(self, addr: int) -> lldb.SBMemoryRegionInfo:
        return self.memory_regions[0]

    def get_thread_with_id(self, tid: int):
        return {}

    def get_registers_for_thread(self, tid: int):
        return {}

    def read_memory_at_address(self, addr: int, size: int) -> lldb.SBData:
        data = lldb.SBData().CreateDataFromCString(
                                    self.target.GetByteOrder(),
                                    self.target.GetCodeByteSize(),
                                    "Hello, world!")
        return data

    def get_loaded_images(self):
        return self.loaded_images

    def get_process_id(self) -> int:
        return 42

    def should_stop(self) -> bool:
        return True

    def is_alive(self) -> bool:
        return True

def __lldb_init_module(debugger, dict):
    if not 'SKIP_SCRIPTED_PROCESS_LAUNCH' in os.environ:
        debugger.HandleCommand(
            "process launch -C %s.%s" % (__name__,
                                     MyScriptedProcess.__name__))
    else:
        print("Name of the class that will manage the scripted process: '%s.%s'"
                % (__name__, MyScriptedProcess.__name__))