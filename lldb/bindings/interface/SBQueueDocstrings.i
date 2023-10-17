%feature("docstring",
"Represents a libdispatch queue in the process."
) lldb::SBQueue;

%feature("autodoc", "
    Returns an lldb::queue_id_t type unique identifier number for this
    queue that will not be used by any other queue during this process'
    execution.  These ID numbers often start at 1 with the first
    system-created queues and increment from there."
) lldb::SBQueue::GetQueueID;

%feature("autodoc", "
    Returns an lldb::QueueKind enumerated value (e.g. eQueueKindUnknown,
    eQueueKindSerial, eQueueKindConcurrent) describing the type of this
    queue."
) lldb::SBQueue::GetKind;
