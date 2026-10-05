//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TARGETGROUP_H
#define LLDB_TARGET_TARGETGROUP_H

#include "lldb/lldb-forward.h"

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace lldb_private {

/// Describes the part a target plays in a target group.
enum class TargetRole {
  None = 0,
  CPU,
  Accelerator,
};

/// A group of related targets and the roles they play in that relationship.
/// Groups are created and owned by a Debugger's TargetGroupList. Deleting a
/// group makes any retained shared pointer inactive.
class TargetGroup : public std::enable_shared_from_this<TargetGroup> {
public:
  TargetGroup(const TargetGroup &) = delete;
  TargetGroup &operator=(const TargetGroup &) = delete;
  TargetGroup(TargetGroup &&) = delete;
  TargetGroup &operator=(TargetGroup &&) = delete;

  /// Add \p target_sp to the group with \p role. If the target is already a
  /// member, replace its role. Return false if the target cannot be added.
  bool AddTarget(const lldb::TargetSP &target_sp, TargetRole role);

  /// Remove \p target_sp from the group.
  bool RemoveTarget(const lldb::TargetSP &target_sp);

  /// Return the role assigned to \p target_sp, or TargetRole::None when it is
  /// not a member.
  TargetRole GetRole(const lldb::TargetSP &target_sp) const;

  /// Return live targets whose role is \p role. Passing
  /// TargetRole::None returns every live target in the group.
  std::vector<lldb::TargetSP>
  GetTargets(TargetRole role = TargetRole::None) const;

private:
  friend class TargetGroupList;

  struct Entry {
    lldb::TargetWP target_wp;
    TargetRole role = TargetRole::None;
  };

  TargetGroup(Debugger &debugger, std::shared_ptr<std::recursive_mutex> mutex)
      : m_debugger(&debugger), m_mutex(std::move(mutex)) {}

  void Deactivate();

  Debugger *m_debugger;
  std::shared_ptr<std::recursive_mutex> m_mutex;
  std::vector<Entry> m_entries;
  bool m_active = true;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TARGETGROUP_H
