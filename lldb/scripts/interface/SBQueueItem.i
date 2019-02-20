//===-- SWIG Interface for SBQueueItem.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBQueueItem
{
public:
    SBQueueItem ();

    SBQueueItem (const lldb::QueueItemSP& queue_item_sp);
    
   ~SBQueueItem();

    bool
    IsValid() const;

    void
    Clear ();

    lldb::QueueItemKind
    GetKind () const;

    void
    SetKind (lldb::QueueItemKind kind);

    lldb::SBAddress
    GetAddress () const;

    void
    SetAddress (lldb::SBAddress addr);

    void
    SetQueueItem (const lldb::QueueItemSP& queue_item_sp);

    lldb::SBThread
    GetExtendedBacktraceThread (const char *type);
};

} // namespace lldb
