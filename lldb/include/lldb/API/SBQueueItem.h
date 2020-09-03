//===-- SBQueueItem.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBQUEUEITEM_H
#define LLDB_API_SBQUEUEITEM_H

#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDefines.h"

namespace lldb {

class LLDB_API SBQueueItem {
public:
  SBQueueItem();

  SBQueueItem(const lldb::QueueItemSP &queue_item_sp);

  ~SBQueueItem();

  explicit operator bool() const;

  bool IsValid() const;

  void Clear();

  lldb::QueueItemKind GetKind() const;

  void SetKind(lldb::QueueItemKind kind);

  lldb::SBAddress GetAddress() const;

  void SetAddress(lldb::SBAddress addr);

  void SetQueueItem(const lldb::QueueItemSP &queue_item_sp);

  SBThread GetExtendedBacktraceThread(const char *type);

private:
  lldb::QueueItemSP m_queue_item_sp;
};

} // namespace lldb

#endif // LLDB_API_SBQUEUEITEM_H
