%feature("docstring",
"Represents the environment of a certain process.

Example: ::

  for entry in lldb.debugger.GetSelectedTarget().GetEnvironment().GetEntries():
    print(entry)

") lldb::SBEnvironment;
