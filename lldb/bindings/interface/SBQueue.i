//===-- SWIG Interface for SBQueue.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a libdispatch queue in the process."
) SBQueue;
class SBQueue
{
public:
    SBQueue ();

    SBQueue (const lldb::QueueSP& queue_sp);

   ~SBQueue();

    bool
    IsValid() const;

    explicit operator bool() const;

    void
    Clear ();

    lldb::SBProcess
    GetProcess ();

    %feature("autodoc", "
    Returns an lldb::queue_id_t type unique identifier number for this
    queue that will not be used by any other queue during this process'
    execution.  These ID numbers often start at 1 with the first
    system-created queues and increment from there.")
    GetQueueID;

    lldb::queue_id_t
    GetQueueID () const;

    const char *
    GetName () const;

    %feature("autodoc", "
    Returns an lldb::QueueKind enumerated value (e.g. eQueueKindUnknown,
    eQueueKindSerial, eQueueKindConcurrent) describing the type of this
    queue.")
    GetKind();

    lldb::QueueKind
    GetKind();

    uint32_t
    GetIndexID () const;

    uint32_t
    GetNumThreads ();

    lldb::SBThread
    GetThreadAtIndex (uint32_t);

    uint32_t
    GetNumPendingItems ();

    lldb::SBQueueItem
    GetPendingItemAtIndex (uint32_t);

    uint32_t
    GetNumRunningItems ();

};

} // namespace lldb

