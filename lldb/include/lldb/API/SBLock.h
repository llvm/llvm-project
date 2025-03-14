//===-- SBLock.h ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBLOCK_H
#define LLDB_API_SBLOCK_H

#include "lldb/API/SBDefines.h"
#include "lldb/lldb-forward.h"

#include <mutex>

namespace lldb_private {
struct APILock;
}

namespace lldb {

#ifndef SWIG
class LLDB_API SBLock {
public:
  ~SBLock();

  bool IsValid() const;

private:
  SBLock() = default;

  // Private constructor used by SBTarget to create the Target API lock.
  // Requires a friend declaration.
  SBLock(std::recursive_mutex &mutex, lldb::TargetSP target_sp);
  friend class SBTarget;

  SBLock(const SBLock &rhs) = delete;
  const SBLock &operator=(const SBLock &rhs) = delete;

  std::unique_ptr<lldb_private::APILock> m_opaque_up;
};
#endif

} // namespace lldb

#endif
