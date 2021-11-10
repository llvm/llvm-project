import os,struct,signal

from typing import Any, Dict

import lldb
from lldb.plugins.scripted_process import ScriptedProcess
from lldb.plugins.scripted_process import ScriptedThread

class MyScriptedProcess(ScriptedProcess):
    memory_regions = [
        lldb.SBMemoryRegionInfo("stack", 0x1040b2000, 0x1040b4000, 0b110, True,
                                True)
    ]

    stack_memory_dump = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'main.stack-dump')

    def __init__(self, target: lldb.SBTarget, args : lldb.SBStructuredData):
        super().__init__(target, args)

    def get_memory_region_containing_address(self, addr: int) -> lldb.SBMemoryRegionInfo:
        for region in self.memory_regions:
            if region.GetRegionBase() <= addr < region.GetRegionEnd():
                return region
        return None

    def get_thread_with_id(self, tid: int):
        return {}

    def get_registers_for_thread(self, tid: int):
        return {}

    def read_memory_at_address(self, addr: int, size: int) -> lldb.SBData:
        data = lldb.SBData()

        with open(self.stack_memory_dump, 'rb') as f:
            stack_mem = f.read(-1)
            if not stack_mem:
                return data

            mem_region = self.get_memory_region_containing_address(addr)

            if not mem_region or addr + size > mem_region.GetRegionEnd():
                return data

            offset = addr - mem_region.GetRegionBase()
            shrunk_stack_mem = stack_mem[offset:offset + size]

            error = lldb.SBError()
            data.SetData(error, shrunk_stack_mem,
                                    self.target.GetByteOrder(),
                                    self.target.GetAddressByteSize())
        return data

    def get_loaded_images(self):
        return self.loaded_images

    def get_process_id(self) -> int:
        return 42

    def should_stop(self) -> bool:
        return True

    def is_alive(self) -> bool:
        return True

    def get_scripted_thread_plugin(self):
        return MyScriptedThread.__module__ + "." + MyScriptedThread.__name__


class MyScriptedThread(ScriptedThread):
    register_ctx = {
        "rax":0x00000000000006e4,
        "rbx":0x00000001040b6060,
        "rcx":0x00000001040b2e00,
        "rdx":0x00000001040b2ba8,
        "rdi":0x000000000000002a,
        "rsi":0x00000001040b2b98,
        "rbp":0x00000001040b2a20,
        "rsp":0x00000001040b2a20,
        "r8":0x00000000003e131e,
        "r9":0xffffffff00000000,
        "r10":0x0000000000000000,
        "r11":0x0000000000000246,
        "r12":0x000000010007c3a0,
        "r13":0x00000001040b2b18,
        "r14":0x0000000100003f90,
        "r15":0x00000001040b2b88,
        "rip":0x0000000100003f61,
        "rflags":0x0000000000000206,
        "cs":0x000000000000002b,
        "fs":0x0000000000000000,
        "gs":0x0000000000000000,
    }

    def __init__(self, process, args):
        super().__init__(process, args)

    def get_thread_id(self) -> int:
        return 0x19

    def get_name(self) -> str:
        return MyScriptedThread.__name__ + ".thread-1"

    def get_stop_reason(self) -> Dict[str, Any]:
        return { "type": lldb.eStopReasonSignal, "data": {
            "signal": signal.SIGINT
        } }

    def get_stackframes(self):
        class ScriptedStackFrame:
            def __init__(idx, cfa, pc, symbol_ctx):
                self.idx = idx
                self.cfa = cfa
                self.pc = pc
                self.symbol_ctx = symbol_ctx


        symbol_ctx = lldb.SBSymbolContext()
        frame_zero = ScriptedStackFrame(0, 0x42424242, 0x5000000, symbol_ctx)
        self.frames.append(frame_zero)

        return self.frame_zero[0:0]

    def get_register_context(self) -> str:
        return struct.pack("{}Q".format(len(self.register_ctx)), *self.register_ctx.values())


def __lldb_init_module(debugger, dict):
    if not 'SKIP_SCRIPTED_PROCESS_LAUNCH' in os.environ:
        debugger.HandleCommand(
            "process launch -C %s.%s" % (__name__,
                                     MyScriptedProcess.__name__))
    else:
        print("Name of the class that will manage the scripted process: '%s.%s'"
                % (__name__, MyScriptedProcess.__name__))