%feature("docstring",
"Represents a container for holding any error code.

For example (from test/python_api/hello_world/TestHelloWorld.py), ::

    def hello_world_attach_with_id_api(self):
        '''Create target, spawn a process, and attach to it by id.'''

        target = self.dbg.CreateTarget(self.exe)

        # Spawn a new process and don't display the stdout if not in TraceOn() mode.
        import subprocess
        popen = subprocess.Popen(
            [self.exe, 'abc', 'xyz'],
            stdout=subprocess.DEVNULL if not self.TraceOn() else None,
        )

        listener = lldb.SBListener('my.attach.listener')
        error = lldb.SBError()
        process = target.AttachToProcessWithID(listener, popen.pid, error)

        self.assertTrue(error.Success() and process, PROCESS_IS_VALID)

        # Let's check the stack traces of the attached process.
        import lldbutil
        stacktraces = lldbutil.print_stacktraces(process, string_buffer=True)
        self.expect(stacktraces, exe=False,
            substrs = ['main.c:%d' % self.line2,
                       '(int)argc=3'])

        listener = lldb.SBListener('my.attach.listener')
        error = lldb.SBError()
        process = target.AttachToProcessWithID(listener, popen.pid, error)

        self.assertTrue(error.Success() and process, PROCESS_IS_VALID)

checks that after the attach, there is no error condition by asserting
that error.Success() is True and we get back a valid process object.

And (from test/python_api/event/TestEvent.py), ::

        # Now launch the process, and do not stop at entry point.
        error = lldb.SBError()
        process = target.Launch(listener, None, None, None, None, None, None, 0, False, error)
        self.assertTrue(error.Success() and process, PROCESS_IS_VALID)

checks that after calling the target.Launch() method there's no error
condition and we get back a void process object.") lldb::SBError;
