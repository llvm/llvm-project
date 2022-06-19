//===-- UserIDResolver.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_USERIDRESOLVER_H
#define LLDB_UTILITY_USERIDRESOLVER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <mutex>

namespace lldb_private {

/// An abstract interface for things that know how to map numeric user/group IDs
/// into names. It caches the resolved names to avoid repeating expensive
/// queries. The cache is internally protected by a mutex, so concurrent queries
/// are safe.
class UserIDResolver {
public:
  typedef uint32_t id_t;
  virtual ~UserIDResolver(); // anchor

  llvm::Optional<llvm::StringRef> GetUserName(id_t uid) {
    return Get(uid, m_uid_cache, &UserIDResolver::DoGetUserName);
  }
  llvm::Optional<llvm::StringRef> GetGroupName(id_t gid) {
    return Get(gid, m_gid_cache, &UserIDResolver::DoGetGroupName);
  }

  /// Returns a resolver which returns a failure value for each query. Useful as
  /// a fallback value for the case when we know all lookups will fail.
  static UserIDResolver &GetNoopResolver();

protected:
  virtual llvm::Optional<std::string> DoGetUserName(id_t uid) = 0;
  virtual llvm::Optional<std::string> DoGetGroupName(id_t gid) = 0;

private:
  using Map = llvm::DenseMap<id_t, llvm::Optional<std::string>>;

  llvm::Optional<llvm::StringRef>
  Get(id_t id, Map &cache,
      llvm::Optional<std::string> (UserIDResolver::*do_get)(id_t));

  std::mutex m_mutex;
  Map m_uid_cache;
  Map m_gid_cache;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_USERIDRESOLVER_H
