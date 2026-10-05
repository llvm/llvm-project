//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TARGETGROUPLIST_H
#define LLDB_TARGET_TARGETGROUPLIST_H

#include "lldb/Target/TargetGroup.h"

#include <memory>
#include <mutex>
#include <vector>

namespace lldb_private {

/// The target groups owned by one Debugger.
class TargetGroupList {
private:
  friend class Debugger;
  friend class Target;

  explicit TargetGroupList(Debugger &debugger)
      : m_debugger(debugger),
        m_mutex(std::make_shared<std::recursive_mutex>()) {}

public:
  ~TargetGroupList();

  TargetGroupList(const TargetGroupList &) = delete;
  TargetGroupList &operator=(const TargetGroupList &) = delete;
  TargetGroupList(TargetGroupList &&) = delete;
  TargetGroupList &operator=(TargetGroupList &&) = delete;

  lldb::TargetGroupSP CreateTargetGroup();

  bool DeleteTargetGroup(const lldb::TargetGroupSP &group_sp);

  size_t GetNumTargetGroups() const;

  lldb::TargetGroupSP GetTargetGroupAtIndex(size_t index) const;

private:
  void Clear();

  Debugger &m_debugger;
  std::shared_ptr<std::recursive_mutex> m_mutex;
  std::vector<lldb::TargetGroupSP> m_groups;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TARGETGROUPLIST_H
