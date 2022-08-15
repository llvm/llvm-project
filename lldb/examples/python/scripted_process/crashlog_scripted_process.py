import os,json,struct,signal,uuid

from typing import Any, Dict

import lldb
from lldb.plugins.scripted_process import ScriptedProcess
from lldb.plugins.scripted_process import ScriptedThread

from lldb.macosx.crashlog import CrashLog,CrashLogParser

class CrashLogScriptedProcess(ScriptedProcess):
    def parse_crashlog(self):
        crashlog_parser = CrashLogParser(self.dbg, self.crashlog_path, False)
        crash_log = crashlog_parser.parse()

        self.pid = crash_log.process_id
        self.addr_mask = crash_log.addr_mask
        self.crashed_thread_idx = crash_log.crashed_thread_idx
        self.loaded_images = []
        self.exception = crash_log.exception

        def load_images(self, images):
            #TODO: Add to self.loaded_images and load images in lldb
            if images:
                for image in images:
                    if image not in self.loaded_images:
                        if image.uuid == uuid.UUID(int=0):
                            continue
                        err = image.add_module(self.target)
                        if err:
                            # Append to SBCommandReturnObject
                            print(err)
                        else:
                            self.loaded_images.append(image)

        for thread in crash_log.threads:
            if self.load_all_images:
                load_images(self, crash_log.images)
            elif thread.did_crash():
                for ident in thread.idents:
                    load_images(self, crash_log.find_images_with_identifier(ident))

            self.threads[thread.index] = CrashLogScriptedThread(self, None, thread)

    def __init__(self, target: lldb.SBTarget, args : lldb.SBStructuredData):
        super().__init__(target, args)

        if not self.target or not self.target.IsValid():
            # Return error
            return

        self.crashlog_path = None

        crashlog_path = args.GetValueForKey("crashlog_path")
        if crashlog_path and crashlog_path.IsValid():
            if crashlog_path.GetType() == lldb.eStructuredDataTypeString:
                self.crashlog_path = crashlog_path.GetStringValue(4096)

        if not self.crashlog_path:
            # Return error
            return

        load_all_images = args.GetValueForKey("load_all_images")
        if load_all_images and load_all_images.IsValid():
            if load_all_images.GetType() == lldb.eStructuredDataTypeBoolean:
                self.load_all_images = load_all_images.GetBooleanValue()

        if not self.load_all_images:
            self.load_all_images = False

        self.pid = super().get_process_id()
        self.crashed_thread_idx = 0
        self.exception = None
        self.parse_crashlog()

    def get_memory_region_containing_address(self, addr: int) -> lldb.SBMemoryRegionInfo:
        return None

    def get_thread_with_id(self, tid: int):
        return {}

    def get_registers_for_thread(self, tid: int):
        return {}

    def read_memory_at_address(self, addr: int, size: int) -> lldb.SBData:
        # NOTE: CrashLogs don't contain any memory.
        return lldb.SBData()

    def get_loaded_images(self):
        # TODO: Iterate over corefile_target modules and build a data structure
        # from it.
        return self.loaded_images

    def get_process_id(self) -> int:
        return self.pid

    def should_stop(self) -> bool:
        return True

    def is_alive(self) -> bool:
        return True

    def get_scripted_thread_plugin(self):
        return CrashLogScriptedThread.__module__ + "." + CrashLogScriptedThread.__name__

class CrashLogScriptedThread(ScriptedThread):
    def create_register_ctx(self):
        if not self.has_crashed:
            return dict.fromkeys([*map(lambda reg: reg['name'], self.register_info['registers'])] , 0)

        if not self.backing_thread or not len(self.backing_thread.registers):
            return dict.fromkeys([*map(lambda reg: reg['name'], self.register_info['registers'])] , 0)

        for reg in self.register_info['registers']:
            reg_name = reg['name']
            if reg_name in self.backing_thread.registers:
                self.register_ctx[reg_name] = self.backing_thread.registers[reg_name]
            else:
                self.register_ctx[reg_name] = 0

        return self.register_ctx

    def create_stackframes(self):
        if not (self.scripted_process.load_all_images or self.has_crashed):
            return None

        if not self.backing_thread or not len(self.backing_thread.frames):
            return None

        for frame in self.backing_thread.frames:
            frame_pc = frame.pc & self.scripted_process.addr_mask
            pc = frame_pc if frame.index == 0  or frame_pc == 0 else frame_pc - 1
            sym_addr = lldb.SBAddress()
            sym_addr.SetLoadAddress(pc, self.target)
            if not sym_addr.IsValid():
                continue
            self.frames.append({"idx": frame.index, "pc": pc})

        return self.frames

    def __init__(self, process, args, crashlog_thread):
        super().__init__(process, args)

        self.backing_thread = crashlog_thread
        self.idx = self.backing_thread.index
        self.tid = self.backing_thread.id
        self.name = self.backing_thread.name
        self.queue = self.backing_thread.queue
        self.has_crashed = (self.scripted_process.crashed_thread_idx == self.idx)
        self.create_stackframes()

    def get_state(self):
        if not self.has_crashed:
            return lldb.eStateStopped
        return lldb.eStateCrashed

    def get_stop_reason(self) -> Dict[str, Any]:
        if not self.has_crashed:
            return { "type": lldb.eStopReasonNone }
        # TODO: Investigate what stop reason should be reported when crashed
        stop_reason = { "type": lldb.eStopReasonException, "data": {  }}
        if self.scripted_process.exception:
            stop_reason['data']['mach_exception'] = self.scripted_process.exception
        return stop_reason

    def get_register_context(self) -> str:
        if not self.register_ctx:
            self.register_ctx = self.create_register_ctx()

        return struct.pack("{}Q".format(len(self.register_ctx)), *self.register_ctx.values())
