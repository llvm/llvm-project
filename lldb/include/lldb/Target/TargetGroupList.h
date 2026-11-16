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
#include <type_traits>
#include <utility>
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

  /// Create and register a target group.
  lldb::TargetGroupSP CreateTargetGroup(llvm::StringRef description);

  /// Create and register a target group of type GroupT. GroupT is constructed
  /// with this list, \p description, and \p args. Its constructor may be
  /// private if it friends TargetGroupList. Runtime-identifiable subclasses
  /// should use llvm::RTTIExtends<GroupT, TargetGroup>.
  template <typename GroupT, typename... Args>
  std::shared_ptr<GroupT> CreateTargetGroup(llvm::StringRef description,
                                            Args &&...args) {
    static_assert(std::is_base_of_v<TargetGroup, GroupT>,
                  "GroupT must derive from TargetGroup");
    auto group_sp = std::shared_ptr<GroupT>(
        new GroupT(*this, description, std::forward<Args>(args)...));
    std::lock_guard<std::recursive_mutex> guard(*m_mutex);
    m_groups.push_back(group_sp);
    return group_sp;
  }

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
