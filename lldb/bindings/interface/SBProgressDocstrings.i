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

Below are examples in Python for deterministic and non-deterministic progresses. ::

    deterministic_progress1 = lldb.SBProgress('Deterministic Progress', 'Detail', 3, lldb.SBDebugger) 
    for i in range(3): 
        deterministic_progress1.Increment(1, f'Update {i}') 
    # The call to Finalize() is a no-op as we already incremented the right amount of 
    # times and caused the end event to be sent. 
    deterministic_progress1.Finalize()

    deterministic_progress2 = lldb.SBProgress('Deterministic Progress', 'Detail', 10, lldb.SBDebugger) 
    for i in range(3): 
        deterministic_progress2.Increment(1, f'Update {i}')      
    # Cause the progress end event to be sent even if we didn't increment the right 
    # number of times. Real world examples would be in a try-finally block to ensure
    # progress clean-up.
    deterministic_progress2.Finalize() 

If you don't call Finalize() when the progress is not done, the progress object will eventually get
garbage collected by the Python runtime, the end event will eventually get sent, but it is best not to 
rely on the garbage collection when using lldb.SBProgress.

Non-deterministic progresses behave the same, but omit the total in the constructor. ::

    non_deterministic_progress = lldb.SBProgress('Non deterministic progress', 'Detail', lldb.SBDebugger)
    for i in range(10):
        non_deterministic_progress.Increment(1)
    # Explicitly send a progressEnd, otherwise this will be sent
    # when the python runtime cleans up this object.
    non_deterministic_progress.Finalize()

Additionally for Python, progress is supported in a with statement. ::
    with lldb.SBProgress('Non deterministic progress', 'Detail', lldb.SBDebugger) as progress:
        for i in range(10):
            progress.Increment(1)
            ...

The progress object is automatically finalized on the exit of the with block.
") lldb::SBProgress;    

%feature("docstring",
"Finalize the SBProgress, which will cause a progress end event to be emitted. This 
happens automatically when the SBProcess object is destroyed, but can be done explicitly 
with Finalize to avoid having to rely on the language semantics for destruction.

Note once finalized, no further increments will be processed.") lldb::SBProgress::Finalize;

%feature("docstring",
"Increment the progress by a given number of units, optionally with a message. Not all progress events are guaraunteed
to be sent, but incrementing to the total will always guarauntee a progress end event being sent.") lldb::SBProcess::Increment; 
