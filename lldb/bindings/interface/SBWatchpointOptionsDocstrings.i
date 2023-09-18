%feature("docstring",
"A container for options to use when creating watchpoints."
) lldb::SBWatchpointOptions;

%feature("docstring", "Sets whether the watchpoint should stop on read accesses."
) lldb::SBWatchpointOptions::SetWatchpointTypeRead;
%feature("docstring", "Gets whether the watchpoint should stop on read accesses."
) lldb::SBWatchpointOptions::GetWatchpointTypeRead;
%feature("docstring", "Sets whether the watchpoint should stop on write accesses. eWatchpointWriteTypeOnModify is the most commonly useful mode, where lldb will stop when the watched value has changed. eWatchpointWriteTypeAlways will stop on any write to the watched region, and on some targets there can false watchpoint stops where memory near the watched region was written, and lldb cannot detect that it is a spurious stop."
) lldb::SBWatchpointOptions::SetWatchpointTypeWrite;
%feature("docstring", "Gets whether the watchpoint should stop on write accesses, returning WatchpointWriteType to indicate the type of write watching that is enabled, or eWatchpointWriteTypeDisabled."
) lldb::SBWatchpointOptions::GetWatchpointTypeWrite;
