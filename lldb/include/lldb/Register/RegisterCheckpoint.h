//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Opaque register restore operations tied to the backend that captured them.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_REGISTERCHECKPOINT_H
#define LLDB_REGISTER_REGISTERCHECKPOINT_H

#include "llvm/Support/Error.h"
#include <functional>
#include <memory>

namespace lldb_private {
class RegisterContext;

/// A move-only restore operation bound to the backend owner that captured it.
///
/// Only RegisterContext can create or restore a checkpoint. The owner identity
/// is weak: retaining this object does not by itself keep the owner alive. The
/// operation retains any state needed by its backend, including native images,
/// remote save IDs, and format configuration. Destruction does not restore
/// registers, and restoration errors are reported by the context.
class RegisterCheckpoint {
public:
  RegisterCheckpoint(RegisterCheckpoint &&) = default;
  RegisterCheckpoint &operator=(RegisterCheckpoint &&) = default;
  RegisterCheckpoint(const RegisterCheckpoint &) = delete;
  RegisterCheckpoint &operator=(const RegisterCheckpoint &) = delete;

private:
  friend class RegisterContext;
  RegisterCheckpoint(std::shared_ptr<const void> owner,
                     std::function<llvm::Error()> restore)
      : m_owner(std::move(owner)), m_restore(std::move(restore)) {}
  std::weak_ptr<const void> m_owner;
  std::function<llvm::Error()> m_restore;
};

} // namespace lldb_private
#endif
