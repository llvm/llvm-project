%feature("docstring",
"A Progress indicator helper class.

Any potentially long running sections of code in LLDB should report
progress so that clients are aware of delays that might appear during
debugging. Delays commonly include indexing debug information, parsing
symbol tables for object files, downloading symbols from remote
repositories, and many more things.

The Progress class helps make sure that progress is correctly reported
and will always send an initial progress update, updates when
Progress::Increment() is called, and also will make sure that a progress
completed update is reported even if the user doesn't explicitly cause one
to be sent.") lldb::SBProgress;

%feature("docstring",
"Finalize the SBProgress, which will cause a progress end event to be emitted. This 
happens automatically when the SBProcess object is destroyed, but can be done explicitly 
with Finalize to avoid having to rely on the language semantics for destruction.

Note once finalized, no further increments will be processed.") lldb::SBProgress::Finalize;

%feature("docstring",
"Increment the progress by a given number of units, optionally with a message. Not all progress events are guaraunteed
to be sent, but incrementing to the total will always guarauntee a progress end event being sent.") lldb::SBProcess::Increment; 
