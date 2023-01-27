%feature("docstring",
"Represents an entity which can broadcast events.

A default broadcaster is
associated with an SBCommandInterpreter, SBProcess, and SBTarget.  For
example, use ::

    broadcaster = process.GetBroadcaster()

to retrieve the process's broadcaster.

See also SBEvent for example usage of interacting with a broadcaster."
) lldb::SBBroadcaster;
