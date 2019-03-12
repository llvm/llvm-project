//===-- SBWatchpoint.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBWatchpoint_h_
#define LLDB_SBWatchpoint_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class LLDB_API SBWatchpoint {
public:
  SBWatchpoint();

  SBWatchpoint(const lldb::SBWatchpoint &rhs);

  SBWatchpoint(const lldb::WatchpointSP &wp_sp);

  ~SBWatchpoint();

  const lldb::SBWatchpoint &operator=(const lldb::SBWatchpoint &rhs);

  explicit operator bool() const;

  bool IsValid() const;

  SBError GetError();

  watch_id_t GetID();

  /// With -1 representing an invalid hardware index.
  int32_t GetHardwareIndex();

  lldb::addr_t GetWatchAddress();

  size_t GetWatchSize();

  void SetEnabled(bool enabled);

  bool IsEnabled();

  uint32_t GetHitCount();

  uint32_t GetIgnoreCount();

  void SetIgnoreCount(uint32_t n);

  const char *GetCondition();

  void SetCondition(const char *condition);

  bool GetDescription(lldb::SBStream &description, DescriptionLevel level);

  void Clear();

  lldb::WatchpointSP GetSP() const;

  void SetSP(const lldb::WatchpointSP &sp);

  static bool EventIsWatchpointEvent(const lldb::SBEvent &event);

  static lldb::WatchpointEventType
  GetWatchpointEventTypeFromEvent(const lldb::SBEvent &event);

  static lldb::SBWatchpoint GetWatchpointFromEvent(const lldb::SBEvent &event);

private:
  friend class SBTarget;
  friend class SBValue;

  std::weak_ptr<lldb_private::Watchpoint> m_opaque_wp;
};

} // namespace lldb

#endif // LLDB_SBWatchpoint_h_
