import os
import time
from typing import Optional, Callable, Any, List, Union
import uuid

import dap_server
from dap_server import Source
from lldbsuite.test.decorators import skipIf
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbplatformutil
import lldbgdbserverutils
import base64


# DAP tests as a whole have been flakey on the Windows on Arm bot. See:
# https://github.com/llvm/llvm-project/issues/137660
@skipIf(oslist=["windows"], archs=["aarch64"])
class DAPTestCaseBase(TestBase):
    # set timeout based on whether ASAN was enabled or not. Increase
    # timeout by a factor of 10 if ASAN is enabled.
    DEFAULT_TIMEOUT = dap_server.DEFAULT_TIMEOUT
    NO_DEBUG_INFO_TESTCASE = True

    def create_debug_adapter(
        self,
        lldbDAPEnv: Optional[dict[str, str]] = None,
        connection: Optional[str] = None,
        additional_args: Optional[list[str]] = None,
    ):
        """Create the Visual Studio Code debug adapter"""
        self.assertTrue(
            is_exe(self.lldbDAPExec), "lldb-dap must exist and be executable"
        )
        log_file_path = self.getBuildArtifact("dap.txt")
        self.dap_server = dap_server.DebugAdapterServer(
            executable=self.lldbDAPExec,
            connection=connection,
            init_commands=self.setUpCommands(),
            log_file=log_file_path,
            env=lldbDAPEnv,
            additional_args=additional_args or [],
        )

    def build_and_create_debug_adapter(
        self,
        lldbDAPEnv: Optional[dict[str, str]] = None,
        dictionary: Optional[dict] = None,
        additional_args: Optional[list[str]] = None,
    ):
        self.build(dictionary=dictionary)
        self.create_debug_adapter(lldbDAPEnv, additional_args=additional_args)

    def build_and_create_debug_adapter_for_attach(self):
        """Variant of build_and_create_debug_adapter that builds a uniquely
        named binary."""
        unique_name = str(uuid.uuid4())
        self.build_and_create_debug_adapter(dictionary={"EXE": unique_name})
        return self.getBuildArtifact(unique_name)

    def set_source_breakpoints(
        self, source_path, lines, data=None, wait_for_resolve=True
    ):
        """Sets source breakpoints and returns an array of strings containing
        the breakpoint IDs ("1", "2") for each breakpoint that was set.
        Parameter data is array of data objects for breakpoints.
        Each object in data is 1:1 mapping with the entry in lines.
        It contains optional location/hitCondition/logMessage parameters.
        """
        return self.set_source_breakpoints_from_source(
            Source(path=source_path), lines, data, wait_for_resolve
        )

    def set_source_breakpoints_assembly(
        self, source_reference, lines, data=None, wait_for_resolve=True
    ):
        return self.set_source_breakpoints_from_source(
            Source.build(source_reference=source_reference),
            lines,
            data,
            wait_for_resolve,
        )

    def set_source_breakpoints_from_source(
        self, source: Source, lines, data=None, wait_for_resolve=True
    ):
        response = self.dap_server.request_setBreakpoints(
            source,
            lines,
            data,
        )
        if response is None:
            return []
        breakpoints = response["body"]["breakpoints"]
        breakpoint_ids = []
        for breakpoint in breakpoints:
            breakpoint_ids.append("%i" % (breakpoint["id"]))
        if wait_for_resolve:
            self.wait_for_breakpoints_to_resolve(breakpoint_ids)
        return breakpoint_ids

    def set_function_breakpoints(
        self, functions, condition=None, hitCondition=None, wait_for_resolve=True
    ):
        """Sets breakpoints by function name given an array of function names
        and returns an array of strings containing the breakpoint IDs
        ("1", "2") for each breakpoint that was set.
        """
        response = self.dap_server.request_setFunctionBreakpoints(
            functions, condition=condition, hitCondition=hitCondition
        )
        if response is None:
            return []
        breakpoints = response["body"]["breakpoints"]
        breakpoint_ids = []
        for breakpoint in breakpoints:
            breakpoint_ids.append("%i" % (breakpoint["id"]))
        if wait_for_resolve:
            self.wait_for_breakpoints_to_resolve(breakpoint_ids)
        return breakpoint_ids

    def wait_for_breakpoints_to_resolve(self, breakpoint_ids: list[str]):
        unresolved_breakpoints = self.dap_server.wait_for_breakpoints_to_be_verified(
            breakpoint_ids
        )
        self.assertEqual(
            len(unresolved_breakpoints),
            0,
            f"Expected to resolve all breakpoints. Unresolved breakpoint ids: {unresolved_breakpoints}",
        )

    def wait_until(
        self,
        predicate: Callable[[], bool],
        delay: float = 0.5,
    ) -> bool:
        """Repeatedly run the predicate until either the predicate returns True
        or a timeout has occurred."""
        deadline = time.monotonic() + self.DEFAULT_TIMEOUT
        while deadline > time.monotonic():
            if predicate():
                return True
            time.sleep(delay)
        return False

    def assertCapabilityIsSet(self, key: str, msg: Optional[str] = None) -> None:
        """Assert that given capability is set in the client."""
        self.assertIn(key, self.dap_server.capabilities, msg)
        self.assertEqual(self.dap_server.capabilities[key], True, msg)

    def assertCapabilityIsNotSet(self, key: str, msg: Optional[str] = None) -> None:
        """Assert that given capability is not set in the client."""
        if key in self.dap_server.capabilities:
            self.assertEqual(self.dap_server.capabilities[key], False, msg)

    def verify_breakpoint_hit(self, breakpoint_ids: List[Union[int, str]]):
        """Wait for the process we are debugging to stop, and verify we hit
        any breakpoint location in the "breakpoint_ids" array.
        "breakpoint_ids" should be a list of breakpoint ID strings
        (["1", "2"]). The return value from self.set_source_breakpoints()
        or self.set_function_breakpoints() can be passed to this function"""
        stopped_events = self.dap_server.wait_for_stopped()
        normalized_bp_ids = [str(b) for b in breakpoint_ids]
        for stopped_event in stopped_events:
            if "body" in stopped_event:
                body = stopped_event["body"]
                if "reason" not in body:
                    continue
                if (
                    body["reason"] != "breakpoint"
                    and body["reason"] != "instruction breakpoint"
                ):
                    continue
                if "hitBreakpointIds" not in body:
                    continue
                hit_breakpoint_ids = body["hitBreakpointIds"]
                for bp in hit_breakpoint_ids:
                    if str(bp) in normalized_bp_ids:
                        return
        self.assertTrue(
            False,
            f"breakpoint not hit, wanted breakpoint_ids {breakpoint_ids} in stopped_events {stopped_events}",
        )

    def verify_all_breakpoints_hit(self, breakpoint_ids):
        """Wait for the process we are debugging to stop, and verify we hit
        all of the breakpoint locations in the "breakpoint_ids" array.
        "breakpoint_ids" should be a list of int breakpoint IDs ([1, 2])."""
        stopped_events = self.dap_server.wait_for_stopped()
        for stopped_event in stopped_events:
            if "body" in stopped_event:
                body = stopped_event["body"]
                if "reason" not in body:
                    continue
                if (
                    body["reason"] != "breakpoint"
                    and body["reason"] != "instruction breakpoint"
                ):
                    continue
                if "hitBreakpointIds" not in body:
                    continue
                hit_bps = body["hitBreakpointIds"]
                if all(breakpoint_id in hit_bps for breakpoint_id in breakpoint_ids):
                    return
        self.assertTrue(False, f"breakpoints not hit, stopped_events={stopped_events}")

    def verify_stop_exception_info(self, expected_description):
        """Wait for the process we are debugging to stop, and verify the stop
        reason is 'exception' and that the description matches
        'expected_description'
        """
        stopped_events = self.dap_server.wait_for_stopped()
        for stopped_event in stopped_events:
            if "body" in stopped_event:
                body = stopped_event["body"]
                if "reason" not in body:
                    continue
                if body["reason"] != "exception":
                    continue
                if "description" not in body:
                    continue
                description = body["description"]
                if expected_description == description:
                    return True
        return False

    def verify_commands(self, flavor: str, output: str, commands: list[str]):
        self.assertTrue(output and len(output) > 0, "expect console output")
        lines = output.splitlines()
        prefix = "(lldb) "
        for cmd in commands:
            found = False
            for line in lines:
                if len(cmd) > 0 and (cmd[0] == "!" or cmd[0] == "?"):
                    cmd = cmd[1:]
                if line.startswith(prefix) and cmd in line:
                    found = True
                    break
            self.assertTrue(
                found,
                f"Command '{flavor}' - '{cmd}' not found in output: {output}",
            )

    def verify_invalidated_event(self, expected_areas):
        event = self.dap_server.invalidated_event
        self.dap_server.invalidated_event = None
        self.assertIsNotNone(event)
        areas = event["body"].get("areas", [])
        self.assertEqual(set(expected_areas), set(areas))

    def verify_memory_event(self, memoryReference):
        if memoryReference is None:
            self.assertIsNone(self.dap_server.memory_event)
        event = self.dap_server.memory_event
        self.dap_server.memory_event = None
        self.assertIsNotNone(event)
        self.assertEqual(memoryReference, event["body"].get("memoryReference"))

    def get_dict_value(self, d: dict, key_path: list[str]) -> Any:
        """Verify each key in the key_path array is in contained in each
        dictionary within "d". Assert if any key isn't in the
        corresponding dictionary. This is handy for grabbing values from VS
        Code response dictionary like getting
        response['body']['stackFrames']
        """
        value = d
        for key in key_path:
            if key in value:
                value = value[key]
            else:
                self.assertTrue(
                    key in value,
                    'key "%s" from key_path "%s" not in "%s"' % (key, key_path, d),
                )
        return value

    def get_stackFrames_and_totalFramesCount(
        self, threadId=None, startFrame=None, levels=None, format=None, dump=False
    ):
        response = self.dap_server.request_stackTrace(
            threadId=threadId,
            startFrame=startFrame,
            levels=levels,
            format=format,
            dump=dump,
        )
        if response:
            stackFrames = self.get_dict_value(response, ["body", "stackFrames"])
            totalFrames = self.get_dict_value(response, ["body", "totalFrames"])
            self.assertTrue(
                totalFrames > 0,
                "verify totalFrames count is provided by extension that supports "
                "async frames loading",
            )
            return (stackFrames, totalFrames)
        return (None, 0)

    def get_stackFrames(
        self, threadId=None, startFrame=None, levels=None, format=None, dump=False
    ):
        (stackFrames, totalFrames) = self.get_stackFrames_and_totalFramesCount(
            threadId=threadId,
            startFrame=startFrame,
            levels=levels,
            format=format,
            dump=dump,
        )
        return stackFrames

    def get_exceptionInfo(self, threadId=None):
        response = self.dap_server.request_exceptionInfo(threadId=threadId)
        return self.get_dict_value(response, ["body"])

    def get_source_and_line(self, threadId=None, frameIndex=0):
        stackFrames = self.get_stackFrames(
            threadId=threadId, startFrame=frameIndex, levels=1
        )
        if stackFrames is not None:
            stackFrame = stackFrames[0]
            ["source", "path"]
            if "source" in stackFrame:
                source = stackFrame["source"]
                if "path" in source:
                    if "line" in stackFrame:
                        return (source["path"], stackFrame["line"])
        return ("", 0)

    def get_stdout(self):
        return self.dap_server.get_output("stdout")

    def get_console(self):
        return self.dap_server.get_output("console")

    def get_important(self):
        return self.dap_server.get_output("important")

    def collect_stdout(self, pattern: Optional[str] = None) -> str:
        return self.dap_server.collect_output("stdout", pattern=pattern)

    def collect_console(self, pattern: Optional[str] = None) -> str:
        return self.dap_server.collect_output("console", pattern=pattern)

    def collect_important(self, pattern: Optional[str] = None) -> str:
        return self.dap_server.collect_output("important", pattern=pattern)

    def get_local_as_int(self, name, threadId=None):
        value = self.dap_server.get_local_variable_value(name, threadId=threadId)
        # 'value' may have the variable value and summary.
        # Extract the variable value since summary can have nonnumeric characters.
        value = value.split(" ")[0]
        if value.startswith("0x"):
            return int(value, 16)
        elif value.startswith("0"):
            return int(value, 8)
        else:
            return int(value)

    def set_variable(self, varRef, name, value, id=None):
        """Set a variable."""
        response = self.dap_server.request_setVariable(varRef, name, str(value), id=id)
        if response["success"]:
            self.verify_invalidated_event(["variables"])
            self.verify_memory_event(response["body"].get("memoryReference"))
        return response

    def set_local(self, name, value, id=None):
        """Set a top level local variable only."""
        return self.set_variable(1, name, str(value), id=id)

    def set_global(self, name, value, id=None):
        """Set a top level global variable only."""
        return self.set_variable(2, name, str(value), id=id)

    def stepIn(
        self,
        threadId=None,
        targetId=None,
        waitForStop=True,
        granularity="statement",
    ):
        response = self.dap_server.request_stepIn(
            threadId=threadId, targetId=targetId, granularity=granularity
        )
        self.assertTrue(response["success"])
        if waitForStop:
            return self.dap_server.wait_for_stopped()
        return None

    def stepOver(
        self,
        threadId=None,
        waitForStop=True,
        granularity="statement",
    ):
        response = self.dap_server.request_next(
            threadId=threadId, granularity=granularity
        )
        self.assertTrue(
            response["success"], f"next request failed: response {response}"
        )
        if waitForStop:
            return self.dap_server.wait_for_stopped()
        return None

    def stepOut(self, threadId=None, waitForStop=True):
        self.dap_server.request_stepOut(threadId=threadId)
        if waitForStop:
            return self.dap_server.wait_for_stopped()
        return None

    def do_continue(self):  # `continue` is a keyword.
        resp = self.dap_server.request_continue()
        self.assertTrue(resp["success"], f"continue request failed: {resp}")

    def continue_to_next_stop(self):
        self.do_continue()
        return self.dap_server.wait_for_stopped()

    def continue_to_breakpoint(self, breakpoint_id: str):
        self.continue_to_breakpoints((breakpoint_id))

    def continue_to_breakpoints(self, breakpoint_ids):
        self.do_continue()
        self.verify_breakpoint_hit(breakpoint_ids)

    def continue_to_exception_breakpoint(self, filter_label):
        self.do_continue()
        self.assertTrue(
            self.verify_stop_exception_info(filter_label),
            'verify we got "%s"' % (filter_label),
        )

    def continue_to_exit(self, exitCode=0):
        self.do_continue()
        stopped_events = self.dap_server.wait_for_stopped()
        self.assertEqual(
            len(stopped_events), 1, "stopped_events = {}".format(stopped_events)
        )
        self.assertEqual(
            stopped_events[0]["event"], "exited", "make sure program ran to completion"
        )
        self.assertEqual(
            stopped_events[0]["body"]["exitCode"],
            exitCode,
            "exitCode == %i" % (exitCode),
        )

    def disassemble(self, threadId=None, frameIndex=None):
        stackFrames = self.get_stackFrames(
            threadId=threadId, startFrame=frameIndex, levels=1
        )
        self.assertIsNotNone(stackFrames)
        memoryReference = stackFrames[0]["instructionPointerReference"]
        self.assertIsNotNone(memoryReference)

        instructions = self.dap_server.request_disassemble(
            memoryReference=memoryReference
        )
        disassembled_instructions = {}
        for inst in instructions:
            disassembled_instructions[inst["address"]] = inst

        return disassembled_instructions, disassembled_instructions[memoryReference]

    def _build_error_message(self, base_message, response):
        """Build a detailed error message from a DAP response.
        Extracts error information from various possible locations in the response structure.
        """
        error_msg = base_message
        if response:
            if "message" in response:
                error_msg += " (%s)" % response["message"]
            elif "body" in response and "error" in response["body"]:
                if "format" in response["body"]["error"]:
                    error_msg += " (%s)" % response["body"]["error"]["format"]
                else:
                    error_msg += " (error in body)"
            else:
                error_msg += " (no error details available)"
        else:
            error_msg += " (no response)"
        return error_msg

    def attach(
        self,
        *,
        disconnectAutomatically=True,
        sourceInitFile=False,
        expectFailure=False,
        **kwargs,
    ):
        """Build the default Makefile target, create the DAP debug adapter,
        and attach to the process.
        """

        # Make sure we disconnect and terminate the DAP debug adapter even
        # if we throw an exception during the test case.
        def cleanup():
            if disconnectAutomatically:
                self.dap_server.request_disconnect(terminateDebuggee=True)
            self.dap_server.terminate()

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)
        # Initialize and launch the program
        self.dap_server.request_initialize(sourceInitFile)
        response = self.dap_server.request_attach(**kwargs)
        if expectFailure:
            return response
        if not (response and response["success"]):
            error_msg = self._build_error_message("attach failed", response)
            self.assertTrue(response and response["success"], error_msg)

    def launch(
        self,
        program=None,
        *,
        sourceInitFile=False,
        disconnectAutomatically=True,
        expectFailure=False,
        **kwargs,
    ):
        """Sending launch request to dap"""

        # Make sure we disconnect and terminate the DAP debug adapter,
        # if we throw an exception during the test case
        def cleanup():
            if disconnectAutomatically:
                self.dap_server.request_disconnect(terminateDebuggee=True)
            self.dap_server.terminate()

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # Initialize and launch the program
        self.dap_server.request_initialize(sourceInitFile)
        response = self.dap_server.request_launch(program, **kwargs)
        if expectFailure:
            return response
        if not (response and response["success"]):
            error_msg = self._build_error_message("launch failed", response)
            self.assertTrue(response and response["success"], error_msg)

    def build_and_launch(
        self,
        program,
        *,
        lldbDAPEnv: Optional[dict[str, str]] = None,
        **kwargs,
    ):
        """Build the default Makefile target, create the DAP debug adapter,
        and launch the process.
        """
        self.build_and_create_debug_adapter(lldbDAPEnv)
        self.assertTrue(os.path.exists(program), "executable must exist")

        return self.launch(program, **kwargs)

    def getBuiltinDebugServerTool(self):
        # Tries to find simulation/lldb-server/gdbserver tool path.
        server_tool = None
        if lldbplatformutil.getPlatform() == "linux":
            server_tool = lldbgdbserverutils.get_lldb_server_exe()
            if server_tool is None:
                self.dap_server.request_disconnect(terminateDebuggee=True)
                self.assertIsNotNone(server_tool, "lldb-server not found.")
        elif lldbplatformutil.getPlatform() == "macosx":
            server_tool = lldbgdbserverutils.get_debugserver_exe()
            if server_tool is None:
                self.dap_server.request_disconnect(terminateDebuggee=True)
                self.assertIsNotNone(server_tool, "debugserver not found.")
        return server_tool

    def writeMemory(self, memoryReference, data=None, offset=0, allowPartial=False):
        # This function accepts data in decimal and hexadecimal format,
        # converts it to a Base64 string, and send it to the DAP,
        # which expects Base64 encoded data.
        encodedData = (
            ""
            if data is None
            else base64.b64encode(
                # (bit_length + 7 (rounding up to nearest byte) ) //8 = converts to bytes.
                data.to_bytes((data.bit_length() + 7) // 8, "little")
            ).decode()
        )
        response = self.dap_server.request_writeMemory(
            memoryReference, encodedData, offset=offset, allowPartial=allowPartial
        )
        if response["success"]:
            self.verify_invalidated_event(["all"])
        return response
