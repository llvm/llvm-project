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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lldb_private {

/// A group of related targets and the roles they play in that relationship.
///
/// Target::Destroy() automatically removes a target from all its groups;
/// clients do not need to remove memberships during target destruction.
class TargetGroup : public std::enable_shared_from_this<TargetGroup> {
public:
  /// Describes the part a target plays in this group.
  enum class Role {
    CPU,
    Accelerator,
  };

  /// Target groups have identity and maintain backlinks in their member
  /// targets, so they cannot be copied or moved. Create one with
  /// Debugger::GetTargetGroupList().CreateTargetGroup().
  TargetGroup(const TargetGroup &) = delete;
  TargetGroup &operator=(const TargetGroup &) = delete;
  TargetGroup(TargetGroup &&) = delete;
  TargetGroup &operator=(TargetGroup &&) = delete;
  ~TargetGroup();

  llvm::StringRef GetDescription() const { return m_description; }

  /// Add \p target_sp to the group with \p role. Adding an existing member
  /// succeeds only when its role matches \p role.
  llvm::Error AddTarget(const lldb::TargetSP &target_sp, Role role);

  /// Remove \p target_sp from the group.
  bool RemoveTarget(const lldb::TargetSP &target_sp);

  /// Return the role assigned to \p target_sp, if it is a member.
  std::optional<Role> GetRole(const lldb::TargetSP &target_sp) const;

  /// Return every target in the group.
  std::vector<lldb::TargetSP> GetTargets() const;

  /// Return targets whose role is \p role.
  std::vector<lldb::TargetSP> GetTargets(Role role) const;

private:
  friend class TargetGroupList;

  struct Entry {
    lldb::TargetWP target_wp;
    Role role;
  };

  TargetGroup(const lldb::DebuggerSP &debugger_sp, llvm::StringRef description,
              std::shared_ptr<std::recursive_mutex> mutex)
      : m_debugger_wp(debugger_sp), m_description(description),
        m_mutex(std::move(mutex)) {}

  lldb::DebuggerWP m_debugger_wp;
  std::string m_description;
  std::shared_ptr<std::recursive_mutex> m_mutex;
  /// Targets are weakly held so membership does not extend their lifetime.
  std::vector<Entry> m_entries;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TARGETGROUP_H
