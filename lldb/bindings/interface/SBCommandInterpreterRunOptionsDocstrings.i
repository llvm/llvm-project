%feature("docstring",
"SBCommandInterpreterRunOptions controls how the RunCommandInterpreter runs the code it is fed.

A default SBCommandInterpreterRunOptions object has:

* StopOnContinue: false
* StopOnError:    false
* StopOnCrash:    false
* EchoCommands:   true
* PrintResults:   true
* PrintErrors:    true
* AddToHistory:   true
* AllowRepeats    false

Interactive debug sessions always allow repeats, the AllowRepeats
run option only affects non-interactive sessions.
") lldb::SBCommandInterpreterRunOptions;
