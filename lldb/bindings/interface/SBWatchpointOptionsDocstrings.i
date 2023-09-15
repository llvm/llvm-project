%feature("docstring",
"A container for options to use when creating watchpoints."
) lldb::SBWatchpointOptions;

%feature("docstring", "Sets whether the watchpoint should stop on read accesses."
) lldb::SBWatchpointOptions::SetWatchpointTypeRead;
%feature("docstring", "Gets whether the watchpoint should stop on read accesses."
) lldb::SBWatchpointOptions::GetWatchpointTypeRead;
%feature("docstring", "Sets whether the watchpoint should stop on write accesses."
) lldb::SBWatchpointOptions::SetWatchpointTypeWrite;
%feature("docstring", "Gets whether the watchpoint should stop on write accesses."
) lldb::SBWatchpointOptions::GetWatchpointTypeWrite;
%feature("docstring", "Sets whether the watchpoint should stop on modify accesses."
) lldb::SBWatchpointOptions::SetWatchpointTypeModify;
%feature("docstring", "Gets whether the watchpoint should stop on modify accesses."
) lldb::SBWatchpointOptions::GetWatchpointTypeModify;
