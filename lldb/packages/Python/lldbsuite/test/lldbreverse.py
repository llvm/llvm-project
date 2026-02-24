import os
import os.path
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbproxy import *
import lldbgdbserverutils
import re


class ThreadSnapshot:
    def __init__(self, thread_id, registers):
        self.thread_id = thread_id
        self.registers = registers


class MemoryBlockSnapshot:
    def __init__(self, address, data):
        self.address = address
        self.data = data


class StateSnapshot:
    def __init__(self, thread_snapshots, memory):
        self.thread_snapshots = thread_snapshots
        self.memory = memory
        self.thread_id = None


class RegisterInfo:
    def __init__(self, lldb_index, name, bitsize, little_endian):
        self.lldb_index = lldb_index
        self.name = name
        self.bitsize = bitsize
        self.little_endian = little_endian


BELOW_STACK_POINTER = 16384
ABOVE_STACK_POINTER = 4096

BLOCK_SIZE = 1024

SOFTWARE_BREAKPOINTS = 0
HARDWARE_BREAKPOINTS = 1
WRITE_WATCHPOINTS = 2


class ReverseTestBase(GDBProxyTestBase):
    """
    Base class for tests that need reverse execution.

    This class uses a gdbserver proxy to add very limited reverse-
    execution capability to lldb-server/debugserver for testing
    purposes only.

    To use this class, run the inferior forward until some stopping point.
    Then call `start_recording()` and execute forward again until reaching
    a software breakpoint; this class records the state before each execution executes.
    At that point, the server will accept "bc" and "bs" packets to step
    backwards through the state.
    When executing during recording, we only allow single-step and continue without
    delivering a signal, and only software breakpoint stops are allowed.

    We assume that while recording is enabled, the only effects of instructions
    are on general-purpose registers (read/written by the 'g' and 'G' packets)
    and on memory bytes between [SP - BELOW_STACK_POINTER, SP + ABOVE_STACK_POINTER).
    """

    NO_DEBUG_INFO_TESTCASE = True

    """
    A list of StateSnapshots in time order.

    There is one snapshot per single-stepped instruction,
    representing the state before that instruction was
    executed. The last snapshot in the list is the
    snapshot before the last instruction was executed.
    This is an undo log; we snapshot a superset of the state that may have
    been changed by the instruction's execution.
    """
    snapshots = None
    recording_enabled = False

    breakpoints = None

    pc_register_info = None
    sp_register_info = None
    general_purpose_register_info = None

    def __init__(self, *args, **kwargs):
        GDBProxyTestBase.__init__(self, *args, **kwargs)
        self.breakpoints = [set(), set(), set(), set(), set()]

    def respond(self, packet):
        if not packet:
            raise ValueError("Invalid empty packet")
        if packet == self.server.PACKET_INTERRUPT:
            # Don't send a response. We'll just run to completion.
            return []
        if self.is_command(packet, "qSupported", ":"):
            # Disable multiprocess support in the server and in LLDB
            # since Mac debugserver doesn't support it and we want lldb-server to
            # be consistent with that
            reply = self.pass_through(packet.replace(";multiprocess", ""))
            return reply.replace(";multiprocess", "") + ";ReverseStep+;ReverseContinue+"
        if packet == "c" or packet == "s":
            packet = "vCont;" + packet
        elif (
            packet[0] == "c" or packet[0] == "s" or packet[0] == "C" or packet[0] == "S"
        ):
            raise ValueError(
                "Old-style continuation packets with address or signal not supported yet"
            )
        if self.is_command(packet, "vCont", ";"):
            if self.recording_enabled:
                return self.continue_with_recording(packet)
            snapshots = []
        if packet == "bc":
            return self.reverse_continue()
        if packet == "bs":
            return self.reverse_step()
        if packet == "jThreadsInfo":
            # Suppress this because it contains thread stop reasons which we might
            # need to modify, and we don't want to have to implement that.
            return ""
        if packet[0] == "x":
            # Suppress *binary* reads as results starting with "O" can be mistaken for an output packet
            # by the test server code
            return ""
        if packet[0] == "z" or packet[0] == "Z":
            reply = self.pass_through(packet)
            if reply == "OK":
                self.update_breakpoints(packet)
            return reply
        return GDBProxyTestBase.respond(self, packet)

    def start_recording(self):
        self.recording_enabled = True
        self.snapshots = []

    def stop_recording(self):
        """
        Don't record when executing foward.

        Reverse execution is still supported until the next forward continue.
        """
        self.recording_enabled = False

    def is_command(self, packet, cmd, follow_token):
        return packet == cmd or packet[0 : len(cmd) + 1] == cmd + follow_token

    def update_breakpoints(self, packet):
        m = re.match("([zZ])([01234]),([0-9a-f]+),([0-9a-f]+)", packet)
        if m is None:
            raise ValueError("Invalid breakpoint packet: " + packet)
        t = int(m.group(2))
        addr = int(m.group(3), 16)
        kind = int(m.group(4), 16)
        if m.group(1) == "Z":
            self.breakpoints[t].add((addr, kind))
        else:
            self.breakpoints[t].discard((addr, kind))

    def breakpoint_triggered_at(self, pc):
        if any(addr == pc for addr, kind in self.breakpoints[SOFTWARE_BREAKPOINTS]):
            return True
        if any(addr == pc for addr, kind in self.breakpoints[HARDWARE_BREAKPOINTS]):
            return True
        return False

    def watchpoint_triggered(self, new_value_block, current_contents):
        """Returns the address or None."""
        for watch_addr, kind in self.breakpoints[WRITE_WATCHPOINTS]:
            for offset in range(0, kind):
                addr = watch_addr + offset
                if (
                    addr >= new_value_block.address
                    and addr < new_value_block.address + len(new_value_block.data)
                ):
                    index = addr - new_value_block.address
                    if (
                        new_value_block.data[index * 2 : (index + 1) * 2]
                        != current_contents[index * 2 : (index + 1) * 2]
                    ):
                        return watch_addr
        return None

    def continue_with_recording(self, packet):
        self.logger.debug("Continue with recording enabled")

        step_packet = "vCont;s"
        if packet == "vCont":
            requested_step = False
        else:
            m = re.match("vCont;(c|s)(.*)", packet)
            if m is None:
                raise ValueError("Unsupported vCont packet: " + packet)
            requested_step = m.group(1) == "s"
            step_packet += m.group(2)

        while True:
            snapshot = self.capture_snapshot()
            reply = self.pass_through(step_packet)
            (stop_signal, stop_pairs) = self.parse_stop_reply(reply)
            if stop_signal != 5:
                raise ValueError("Unexpected stop signal: " + reply)
            is_swbreak = False
            thread_id = None
            for key, value in stop_pairs.items():
                if key == "thread":
                    thread_id = self.parse_thread_id(value)
                    continue
                if re.match("[0-9a-f]+", key):
                    continue
                if key == "swbreak" or (key == "reason" and value == "breakpoint"):
                    is_swbreak = True
                    continue
                if key == "metype":
                    reason = self.stop_reason_from_mach_exception(stop_pairs)
                    if reason == "breakpoint":
                        is_swbreak = True
                    elif reason != "singlestep":
                        raise ValueError(f"Unsupported stop reason in {reply}")
                    continue
                if key in [
                    "name",
                    "threads",
                    "thread-pcs",
                    "reason",
                    "mecount",
                    "medata",
                    "memory",
                ]:
                    continue
                raise ValueError(f"Unknown stop key '{key}' in {reply}")
            if is_swbreak:
                self.logger.debug("Recording stopped")
                return reply
            if thread_id is None:
                return ValueError("Expected thread ID: " + reply)
            snapshot.thread_id = thread_id
            self.snapshots.append(snapshot)
            if requested_step:
                self.logger.debug("Recording stopped for step")
                return reply

    def stop_reason_from_mach_exception(self, stop_pairs):
        # See StopInfoMachException::CreateStopReasonWithMachException.
        if int(stop_pairs["metype"]) != 6:  # EXC_BREAKPOINT
            raise ValueError(f"Unsupported exception type {value} in {reply}")
        medata = stop_pairs["medata"]
        arch = self.getArchitecture()
        if arch in ["amd64", "i386", "x86_64"]:
            if int(medata[0], 16) == 2:
                return "breakpoint"
            if int(medata[0], 16) == 1 and int(medata[1], 16) == 0:
                return "singlestep"
        elif arch in ["arm64", "arm64e"]:
            if int(medata[0], 16) == 1 and int(medata[1], 16) != 0:
                return "breakpoint"
            elif int(medata[0], 16) == 1 and int(medata[1], 16) == 0:
                return "singlestep"
        else:
            raise ValueError(f"Unsupported architecture '{arch}'")
        raise ValueError(f"Unsupported exception details in {reply}")

    def parse_stop_reply(self, reply):
        if not reply:
            raise ValueError("Invalid empty packet")
        if reply[0] == "T" and len(reply) >= 3:
            result = {}
            for k, v in self.parse_pairs(reply[3:]):
                if k in ["medata", "memory"]:
                    if k in result:
                        result[k].append(v)
                    else:
                        result[k] = [v]
                else:
                    result[k] = v
            return (int(reply[1:3], 16), result)
        raise ValueError("Unsupported stop reply: " + reply)

    def parse_pairs(self, text):
        for pair in text.split(";"):
            if not pair:
                continue
            m = re.match("([^:]+):(.*)", pair)
            if m is None:
                raise ValueError("Invalid pair text: " + text)
            yield (m.group(1), m.group(2))

    def capture_snapshot(self):
        """Snapshot all threads and their stack memories."""
        self.ensure_register_info()
        current_thread = self.get_current_thread()
        thread_snapshots = []
        memory = []
        for thread_id in self.get_thread_list():
            registers = {}
            for index in sorted(self.general_purpose_register_info.keys()):
                reply = self.pass_through(f"p{index:x};thread:{thread_id:x};")
                if reply == "" or reply[0] == "E":
                    # Mac debugserver tells us about registers that it won't let
                    # us actually read. Ignore those registers.
                    self.logger.debug(f"Failed to read register {index:x}")
                    continue
                registers[index] = reply
            thread_snapshot = ThreadSnapshot(thread_id, registers)
            thread_sp = self.get_register(
                self.sp_register_info, thread_snapshot.registers
            )

            # The memory above or below the stack pointer may be mapped, but not
            # both readable and writeable. For example on Arm 32-bit Linux, there
            # is a "[vectors]" mapping above the stack, which can be read but not
            # written to.
            #
            # Therefore, we should limit any reads to the stack region, which we
            # know is readable and writeable.
            region_info = self.get_memory_region_info(thread_sp)
            lower = max(thread_sp - BELOW_STACK_POINTER, region_info["start"])
            upper = min(
                thread_sp + ABOVE_STACK_POINTER,
                region_info["start"] + region_info["size"],
            )

            memory += self.read_memory(lower, upper)
            thread_snapshots.append(thread_snapshot)
        self.set_current_thread(current_thread)
        return StateSnapshot(thread_snapshots, memory)

    def restore_snapshot(self, snapshot):
        """
        Restore the snapshot during reverse execution.

        If this triggers a breakpoint or watchpoint, return the stop reply,
        otherwise None.
        """
        current_thread = self.get_current_thread()
        stop_reasons = []
        for thread_snapshot in snapshot.thread_snapshots:
            thread_id = thread_snapshot.thread_id
            for lldb_index in sorted(thread_snapshot.registers.keys()):
                data = thread_snapshot.registers[lldb_index]
                reply = self.pass_through(
                    f"P{lldb_index:x}={data};thread:{thread_id:x};"
                )
                if reply != "OK":
                    try:
                        reg_name = self.general_purpose_register_info[lldb_index].name
                    except KeyError:
                        reg_name = f"with index {lldb_index}"
                    raise ValueError(f"Can't restore thread register {reg_name}")
            if thread_id == snapshot.thread_id:
                new_pc = self.get_register(
                    self.pc_register_info, thread_snapshot.registers
                )
                if self.breakpoint_triggered_at(new_pc):
                    stop_reasons.append([("reason", "breakpoint")])
        self.set_current_thread(current_thread)
        for block in snapshot.memory:
            current_memory = self.pass_through(
                f"m{block.address:x},{(len(block.data)//2):x}"
            )
            if not current_memory or current_memory[0] == "E":
                raise ValueError("Can't read back memory")
            reply = self.pass_through(
                f"M{block.address:x},{len(block.data)//2:x}:" + block.data
            )
            if reply != "OK":
                raise ValueError(
                    f"Can't restore memory block ranging from 0x{block.address:x} to 0x{block.address+len(block.data):x}."
                )
            watch_addr = self.watchpoint_triggered(block, current_memory)
            if watch_addr is not None:
                stop_reasons.append(
                    [("reason", "watchpoint"), ("watch", f"{watch_addr:x}")]
                )
        if stop_reasons:
            pairs = ";".join(f"{key}:{value}" for key, value in stop_reasons[0])
            return f"T05thread:{snapshot.thread_id:x};{pairs};"
        return None

    def reverse_step(self):
        if not self.snapshots:
            self.logger.debug("Reverse-step at history boundary")
            return self.history_boundary_reply(self.get_current_thread())
        self.logger.debug("Reverse-step started")
        snapshot = self.snapshots.pop()
        stop_reply = self.restore_snapshot(snapshot)
        self.set_current_thread(snapshot.thread_id)
        self.logger.debug("Reverse-step stopped")
        if stop_reply is None:
            return self.singlestep_stop_reply(snapshot.thread_id)
        return stop_reply

    def reverse_continue(self):
        self.logger.debug("Reverse-continue started")
        thread_id = None
        while self.snapshots:
            snapshot = self.snapshots.pop()
            stop_reply = self.restore_snapshot(snapshot)
            thread_id = snapshot.thread_id
            if stop_reply is not None:
                self.set_current_thread(thread_id)
                self.logger.debug("Reverse-continue stopped")
                return stop_reply
        if thread_id is None:
            thread_id = self.get_current_thread()
        else:
            self.set_current_thread(snapshot.thread_id)
        self.logger.debug("Reverse-continue stopped at history boundary")
        return self.history_boundary_reply(thread_id)

    def get_current_thread(self):
        reply = self.pass_through("qC")
        return self.parse_thread_id(reply[2:])

    def parse_thread_id(self, thread_id):
        m = re.match("([0-9a-f]+)", thread_id)
        if m is None:
            raise ValueError("Invalid thread ID: " + thread_id)
        return int(m.group(1), 16)

    def history_boundary_reply(self, thread_id):
        return f"T00thread:{thread_id:x};replaylog:begin;"

    def singlestep_stop_reply(self, thread_id):
        return f"T05thread:{thread_id:x};"

    def set_current_thread(self, thread_id):
        """
        Set current thread in inner gdbserver.
        """
        if thread_id >= 0:
            self.pass_through(f"Hg{thread_id:x}")
            self.pass_through(f"Hc{thread_id:x}")
        else:
            self.pass_through(f"Hc-1")
            self.pass_through(f"Hg-1")

    def get_register(self, register_info, registers):
        if register_info.bitsize % 8 != 0:
            raise ValueError("Register size must be a multiple of 8 bits")
        if register_info.lldb_index not in registers:
            raise ValueError("Register value not captured")
        data = registers[register_info.lldb_index]
        num_bytes = register_info.bitsize // 8
        bytes = []
        for i in range(0, num_bytes):
            bytes.append(int(data[i * 2 : (i + 1) * 2], 16))
        if register_info.little_endian:
            bytes.reverse()
        result = 0
        for byte in bytes:
            result = (result << 8) + byte
        return result

    def get_memory_region_info(self, addr):
        reply = self.pass_through(f"qMemoryRegionInfo:{addr:x}")
        if not reply or reply[0] == "E":
            raise RuntimeError("Failed to get memory region info.")

        # Valid reply looks like:
        # start:fffcf000;size:21000;permissions:rw;flags:;name:5b737461636b5d;
        values = [v for v in reply.strip().split(";") if v]
        region_info = {}
        for value in values:
            key, value = value.split(
                ":",
            )
            region_info[key] = value

        if not ("start" in region_info and "size" in region_info):
            raise RuntimeError("Did not get extent of memory region.")

        region_info["start"] = int(region_info["start"], 16)
        region_info["size"] = int(region_info["size"], 16)

        return region_info

    def read_memory(self, start_addr, end_addr):
        """
        Read a region of memory from the target.

        Some of the addresses may extend into memory we cannot read, skip those.

        Return a list of blocks containing the valid area(s) in the
        requested range.
        """
        regions = []
        start_addr = start_addr - (start_addr % BLOCK_SIZE)
        if end_addr % BLOCK_SIZE > 0:
            end_addr = end_addr - (end_addr % BLOCK_SIZE) + BLOCK_SIZE
        for addr in range(start_addr, end_addr, BLOCK_SIZE):
            reply = self.pass_through(f"m{addr:x},{(BLOCK_SIZE - 1):x}")
            if reply and reply[0] != "E":
                block = MemoryBlockSnapshot(addr, reply)
                regions.append(block)
        return regions

    def ensure_register_info(self):
        if self.general_purpose_register_info is not None:
            return
        reply = self.pass_through("qHostInfo")
        little_endian = any(
            kv == ("endian", "little") for kv in self.parse_pairs(reply)
        )
        self.general_purpose_register_info = {}
        lldb_index = 0
        while True:
            reply = self.pass_through(f"qRegisterInfo{lldb_index:x}")
            if not reply or reply[0] == "E":
                break
            info = {k: v for k, v in self.parse_pairs(reply)}
            reg_info = RegisterInfo(
                lldb_index, info["name"], int(info["bitsize"]), little_endian
            )
            if (
                info["set"] == "General Purpose Registers"
                and not "container-regs" in info
            ):
                self.general_purpose_register_info[lldb_index] = reg_info
            if "generic" in info:
                if info["generic"] == "pc":
                    self.pc_register_info = reg_info
                elif info["generic"] == "sp":
                    self.sp_register_info = reg_info
            lldb_index += 1
        if self.pc_register_info is None or self.sp_register_info is None:
            raise ValueError("Can't find generic pc or sp register")

    def get_thread_list(self):
        threads = []
        reply = self.pass_through("qfThreadInfo")
        while True:
            if not reply:
                raise ValueError("Missing reply packet")
            if reply[0] == "m":
                for id in reply[1:].split(","):
                    threads.append(self.parse_thread_id(id))
            elif reply[0] == "l":
                return threads
            reply = self.pass_through("qsThreadInfo")
