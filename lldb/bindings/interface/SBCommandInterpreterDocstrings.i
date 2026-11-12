%feature("docstring",
"SBCommandInterpreter handles/interprets commands for lldb.

You get the command interpreter from the :py:class:`SBDebugger` instance.

For example (from test/ python_api/interpreter/TestCommandInterpreterAPI.py),::

    def command_interpreter_api(self):
        '''Test the SBCommandInterpreter APIs.'''
        exe = os.path.join(os.getcwd(), 'a.out')

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Retrieve the associated command interpreter from our debugger.
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        # Exercise some APIs....

        self.assertTrue(ci.HasCommands())
        self.assertTrue(ci.HasAliases())
        self.assertTrue(ci.HasAliasOptions())
        self.assertTrue(ci.CommandExists('breakpoint'))
        self.assertTrue(ci.CommandExists('target'))
        self.assertTrue(ci.CommandExists('platform'))
        self.assertTrue(ci.AliasExists('file'))
        self.assertTrue(ci.AliasExists('run'))
        self.assertTrue(ci.AliasExists('bt'))

        res = lldb.SBCommandReturnObject()
        ci.HandleCommand('breakpoint set -f main.c -l %d' % self.line, res)
        self.assertTrue(res.Succeeded())
        ci.HandleCommand('process launch', res)
        self.assertTrue(res.Succeeded())

        process = ci.GetProcess()
        self.assertTrue(process)

        ...

The HandleCommand() instance method takes two args: the command string and
an SBCommandReturnObject instance which encapsulates the result of command
execution.") lldb::SBCommandInterpreter;
