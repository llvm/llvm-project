from __future__ import print_function

from lldbsuite.test.lldbtest import *
import os
import vscode


class VSCodeTestCaseBase(TestBase):

    def create_debug_adaptor(self):
        '''Create the Visual Studio Code debug adaptor'''
        self.assertTrue(os.path.exists(self.lldbVSCodeExec),
                        'lldb-vscode must exist')
        self.vscode = vscode.DebugAdaptor(executable=self.lldbVSCodeExec)

    def build_and_create_debug_adaptor(self):
        self.build()
        self.create_debug_adaptor()

    def set_source_breakpoints(self, source_path, lines, condition=None,
                               hitCondition=None):
        '''Sets source breakpoints and returns an array of strings containing
           the breakpoint location IDs ("1.1", "1.2") for each breakpoint
           that was set.
        '''
        response = self.vscode.request_setBreakpoints(
            source_path, lines, condition=condition, hitCondition=hitCondition)
        if response is None:
            return []
        breakpoints = response['body']['breakpoints']
        breakpoint_ids = []
        for breakpoint in breakpoints:
            response_id = breakpoint['id']
            bp_id = response_id >> 32
            bp_loc_id = response_id & 0xffffffff
            breakpoint_ids.append('%i.%i' % (bp_id, bp_loc_id))
        return breakpoint_ids

    def set_function_breakpoints(self, functions, condition=None,
                                 hitCondition=None):
        '''Sets breakpoints by function name given an array of function names
           and returns an array of strings containing the breakpoint location
           IDs ("1.1", "1.2") for each breakpoint that was set.
        '''
        response = self.vscode.request_setFunctionBreakpoints(
            functions, condition=condition, hitCondition=hitCondition)
        if response is None:
            return []
        breakpoints = response['body']['breakpoints']
        breakpoint_ids = []
        for breakpoint in breakpoints:
            response_id = breakpoint['id']
            bp_id = response_id >> 32
            bp_loc_id = response_id & 0xffffffff
            breakpoint_ids.append('%i.%i' % (bp_id, bp_loc_id))
        return breakpoint_ids

    def verify_breakpoint_hit(self, breakpoint_ids):
        '''Wait for the process we are debugging to stop, and verify we hit
           any breakpoint location in the "breakpoint_ids" array.
           "breakpoint_ids" should be a list of breakpoint location ID strings
           (["1.1", "2.1"]). The return value from
           self.set_source_breakpoints() can be passed to this function'''
        stopped_events = self.vscode.wait_for_stopped()
        for stopped_event in stopped_events:
            if 'body' in stopped_event:
                body = stopped_event['body']
                if 'reason' not in body:
                    continue
                if body['reason'] != 'breakpoint':
                    continue
                if 'description' not in body:
                    continue
                # Description is "breakpoint 1.1", so look for any location id
                # ("1.1") in the description field as verification that one of
                # the breakpoint locations was hit
                description = body['description']
                for breakpoint_id in breakpoint_ids:
                    if breakpoint_id in description:
                        return True
        return False

    def verify_exception_breakpoint_hit(self, filter_label):
        '''Wait for the process we are debugging to stop, and verify the stop
           reason is 'exception' and that the description matches
           'filter_label'
        '''
        stopped_events = self.vscode.wait_for_stopped()
        for stopped_event in stopped_events:
            if 'body' in stopped_event:
                body = stopped_event['body']
                if 'reason' not in body:
                    continue
                if body['reason'] != 'exception':
                    continue
                if 'description' not in body:
                    continue
                description = body['description']
                if filter_label == description:
                    return True
        return False

    def verify_commands(self, flavor, output, commands):
        self.assertTrue(output and len(output) > 0, "expect console output")
        lines = output.splitlines()
        prefix = '(lldb) '
        for cmd in commands:
            found = False
            for line in lines:
                if line.startswith(prefix) and cmd in line:
                    found = True
                    break
            self.assertTrue(found,
                            "verify '%s' found in console output for '%s'" % (
                                cmd, flavor))

    def get_dict_value(self, d, key_path):
        '''Verify each key in the key_path array is in contained in each
           dictionary within "d". Assert if any key isn't in the
           corresponding dictionary. This is handy for grabbing values from VS
           Code response dictionary like getting
           response['body']['stackFrames']
        '''
        value = d
        for key in key_path:
            if key in value:
                value = value[key]
            else:
                self.assertTrue(key in value,
                                'key "%s" from key_path "%s" not in "%s"' % (
                                    key, key_path, d))
        return value

    def get_stackFrames(self, threadId=None, startFrame=None, levels=None,
                        dump=False):
        response = self.vscode.request_stackTrace(threadId=threadId,
                                                  startFrame=startFrame,
                                                  levels=levels,
                                                  dump=dump)
        if response:
            return self.get_dict_value(response, ['body', 'stackFrames'])
        return None

    def get_source_and_line(self, threadId=None, frameIndex=0):
        stackFrames = self.get_stackFrames(threadId=threadId,
                                           startFrame=frameIndex,
                                           levels=1)
        if stackFrames is not None:
            stackFrame = stackFrames[0]
            ['source', 'path']
            if 'source' in stackFrame:
                source = stackFrame['source']
                if 'path' in source:
                    if 'line' in stackFrame:
                        return (source['path'], stackFrame['line'])
        return ('', 0)

    def get_stdout(self, timeout=0.0):
        return self.vscode.get_output('stdout', timeout=timeout)

    def get_console(self, timeout=0.0):
        return self.vscode.get_output('console', timeout=timeout)

    def get_local_as_int(self, name, threadId=None):
        value = self.vscode.get_local_variable_value(name, threadId=threadId)
        if value.startswith('0x'):
            return int(value, 16)
        elif value.startswith('0'):
            return int(value, 8)
        else:
            return int(value)

    def set_local(self, name, value, id=None):
        '''Set a top level local variable only.'''
        return self.vscode.request_setVariable(1, name, str(value), id=id)

    def set_global(self, name, value, id=None):
        '''Set a top level global variable only.'''
        return self.vscode.request_setVariable(2, name, str(value), id=id)

    def stepIn(self, threadId=None, waitForStop=True):
        self.vscode.request_stepIn(threadId=threadId)
        if waitForStop:
            return self.vscode.wait_for_stopped()
        return None

    def stepOver(self, threadId=None, waitForStop=True):
        self.vscode.request_next(threadId=threadId)
        if waitForStop:
            return self.vscode.wait_for_stopped()
        return None

    def stepOut(self, threadId=None, waitForStop=True):
        self.vscode.request_stepOut(threadId=threadId)
        if waitForStop:
            return self.vscode.wait_for_stopped()
        return None

    def continue_to_next_stop(self):
        self.vscode.request_continue()
        return self.vscode.wait_for_stopped()

    def continue_to_breakpoints(self, breakpoint_ids):
        self.vscode.request_continue()
        self.verify_breakpoint_hit(breakpoint_ids)

    def continue_to_exception_breakpoint(self, filter_label):
        self.vscode.request_continue()
        self.assertTrue(self.verify_exception_breakpoint_hit(filter_label),
                        'verify we got "%s"' % (filter_label))

    def continue_to_exit(self, exitCode=0):
        self.vscode.request_continue()
        stopped_events = self.vscode.wait_for_stopped()
        self.assertTrue(len(stopped_events) == 1,
                        "expecting single 'exited' event")
        self.assertTrue(stopped_events[0]['event'] == 'exited',
                        'make sure program ran to completion')
        self.assertTrue(stopped_events[0]['body']['exitCode'] == exitCode,
                        'exitCode == %i' % (exitCode))

    def attach(self, program=None, pid=None, waitFor=None, trace=None,
               initCommands=None, preRunCommands=None, stopCommands=None,
               exitCommands=None, attachCommands=None):
        '''Build the default Makefile target, create the VSCode debug adaptor,
           and attach to the process.
        '''
        # Make sure we disconnect and terminate the VSCode debug adaptor even
        # if we throw an exception during the test case.
        def cleanup():
            self.vscode.request_disconnect(terminateDebuggee=True)
            self.vscode.terminate()

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)
        # Initialize and launch the program
        self.vscode.request_initialize()
        response = self.vscode.request_attach(
            program=program, pid=pid, waitFor=waitFor, trace=trace,
            initCommands=initCommands, preRunCommands=preRunCommands,
            stopCommands=stopCommands, exitCommands=exitCommands,
            attachCommands=attachCommands)
        if not (response and response['success']):
            self.assertTrue(response['success'],
                            'attach failed (%s)' % (response['message']))

    def build_and_launch(self, program, args=None, cwd=None, env=None,
                         stopOnEntry=False, disableASLR=True,
                         disableSTDIO=False, shellExpandArguments=False,
                         trace=False, initCommands=None, preRunCommands=None,
                         stopCommands=None, exitCommands=None,
                         sourcePath=None, debuggerRoot=None):
        '''Build the default Makefile target, create the VSCode debug adaptor,
           and launch the process.
        '''
        self.build_and_create_debug_adaptor()
        self.assertTrue(os.path.exists(program), 'executable must exist')

        # Make sure we disconnect and terminate the VSCode debug adaptor even
        # if we throw an exception during the test case.
        def cleanup():
            self.vscode.request_disconnect(terminateDebuggee=True)
            self.vscode.terminate()

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # Initialize and launch the program
        self.vscode.request_initialize()
        response = self.vscode.request_launch(
            program,
            args=args,
            cwd=cwd,
            env=env,
            stopOnEntry=stopOnEntry,
            disableASLR=disableASLR,
            disableSTDIO=disableSTDIO,
            shellExpandArguments=shellExpandArguments,
            trace=trace,
            initCommands=initCommands,
            preRunCommands=preRunCommands,
            stopCommands=stopCommands,
            exitCommands=exitCommands,
            sourcePath=sourcePath,
            debuggerRoot=debuggerRoot)
        if not (response and response['success']):
            self.assertTrue(response['success'],
                            'launch failed (%s)' % (response['message']))
