//===-- SWIG Interface for SBError ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a container for holding any error code.

For example (from test/python_api/hello_world/TestHelloWorld.py), ::

    def hello_world_attach_with_id_api(self):
        '''Create target, spawn a process, and attach to it by id.'''

        target = self.dbg.CreateTarget(self.exe)

        # Spawn a new process and don't display the stdout if not in TraceOn() mode.
        import subprocess
        popen = subprocess.Popen([self.exe, 'abc', 'xyz'],
                                 stdout = open(os.devnull, 'w') if not self.TraceOn() else None)

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
condition and we get back a void process object.") SBError;

class SBError {
public:
    SBError ();

    SBError (const lldb::SBError &rhs);

    ~SBError();

    const char *
    GetCString () const;

    void
    Clear ();

    bool
    Fail () const;

    bool
    Success () const;

    uint32_t
    GetError () const;

    lldb::ErrorType
    GetType () const;

    void
    SetError (uint32_t err, lldb::ErrorType type);

    void
    SetErrorToErrno ();

    void
    SetErrorToGenericError ();

    void
    SetErrorString (const char *err_str);

    %varargs(3, char *str = NULL) SetErrorStringWithFormat;
    int
    SetErrorStringWithFormat (const char *format, ...);

    bool
    IsValid () const;

    explicit operator bool() const;

    bool
    GetDescription (lldb::SBStream &description);

    STRING_EXTENSION(SBError)

#ifdef SWIGPYTHON
    %pythoncode %{
        value = property(GetError, None, doc='''A read only property that returns the same result as GetError().''')
        fail = property(Fail, None, doc='''A read only property that returns the same result as Fail().''')
        success = property(Success, None, doc='''A read only property that returns the same result as Success().''')
        description = property(GetCString, None, doc='''A read only property that returns the same result as GetCString().''')
        type = property(GetType, None, doc='''A read only property that returns the same result as GetType().''')
    %}
#endif

};

} // namespace lldb
