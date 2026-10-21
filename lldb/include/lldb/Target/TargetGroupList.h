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

/// A thread-safe collection of target groups. A Debugger uses one as its
/// registry, and each Target uses one for the groups it belongs to.
class TargetGroupList {
private:
  friend class Debugger;
  friend class Target;

  explicit TargetGroupList(Debugger &debugger)
      : m_debugger(debugger),
        m_mutex(std::make_shared<std::recursive_mutex>()) {}

  TargetGroupList(Debugger &debugger,
                  const std::shared_ptr<std::recursive_mutex> &mutex)
      : m_debugger(debugger), m_mutex(mutex) {}

public:
  ~TargetGroupList();

  TargetGroupList(const TargetGroupList &) = delete;
  TargetGroupList &operator=(const TargetGroupList &) = delete;
  TargetGroupList(TargetGroupList &&) = delete;
  TargetGroupList &operator=(TargetGroupList &&) = delete;

  lldb::TargetGroupSP CreateTargetGroup(llvm::StringRef description);

  /// Remove \p group_sp from this list. Other shared references, including
  /// target membership lists, remain valid.
  bool DeleteTargetGroup(const lldb::TargetGroupSP &group_sp);

  /// Return a snapshot of the target groups. Modifying the returned vector
  /// does not modify this list.
  std::vector<lldb::TargetGroupSP> GetGroups() const;

private:
  friend class TargetGroup;

  void Add(const lldb::TargetGroupSP &group_sp);
  void Remove(const TargetGroup *group);
  void Clear();
  void RemoveTarget(const lldb::TargetSP &target_sp);

  Debugger &m_debugger;
  /// Shared by the debugger's list, each target's list, and every group.
  /// Group membership operations never acquire Target::m_mutex.
  std::shared_ptr<std::recursive_mutex> m_mutex;
  std::vector<lldb::TargetGroupSP> m_groups;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TARGETGROUPLIST_H
