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
to be sent.

Progress can either be deterministic, incrementing up to a known total or non-deterministic
with an unbounded total. Deterministic is better if you know the items of work in advance, but non-deterministic
exposes a way to update a user during a long running process that work is taking place.

For all progresses the details provided in the constructor will be sent until an increment detail
is provided. This detail will also continue to be broadcasted on any subsequent update that doesn't
specify a new detail. Some implementations differ on throttling updates and this behavior differs primarily
if the progress is deterministic or non-deterministic. For DAP, non-deterministic update messages have a higher
throttling rate than deterministic ones.

Below are examples in Python for deterministic and non-deterministic progresses.

    deterministic_progress = lldb.SBProgress('Deterministic Progress', 'Detail', 3, lldb.SBDebugger)
    for i in range(3):
        deterministic_progress.Increment(1, f'Update {i}')

    non_deterministic_progress = lldb.SBProgress('Non deterministic progress, 'Detail', lldb.SBDebugger)
    for i in range(10):
        non_deterministic_progress.Increment(1)

    # Explicitly send a progressEnd from Python.
    non_deterministic_progress.Finalize()
") lldb::SBProgress;    

%feature("docstring",
"Finalize the SBProgress, which will cause a progress end event to be emitted. This 
happens automatically when the SBProcess object is destroyed, but can be done explicitly 
with Finalize to avoid having to rely on the language semantics for destruction.

Note once finalized, no further increments will be processed.") lldb::SBProgress::Finalize;

%feature("docstring",
"Increment the progress by a given number of units, optionally with a message. Not all progress events are guaraunteed
to be sent, but incrementing to the total will always guarauntee a progress end event being sent.") lldb::SBProcess::Increment; 
