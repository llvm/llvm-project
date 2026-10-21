//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_STRINGPOOL_H
#define LLDB_UTILITY_STRINGPOOL_H

#include "lldb/Utility/ConstString.h"

#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <memory>

namespace lldb_private {

class StringPoolRef;

/// A thread-safe pool of interned, null-terminated strings. A string returned
/// by a pool stays valid for the lifetime of that pool.
class StringPool {
public:
  StringPool();
  ~StringPool();

  StringPool(const StringPool &) = delete;
  StringPool &operator=(const StringPool &) = delete;

  /// The pool backing ConstString. It is never destroyed.
  static StringPool &GetGlobal();

  /// Set up and tear down the system pool, for strings that are not owned by a
  /// Debugger.
  static void Initialize();
  static void Terminate();

  /// The system pool. Only valid between Initialize and Terminate.
  static StringPoolRef GetSystemPool();

  /// Returns the pooled copy of \p str, or nullptr if \p str has no data.
  const char *Intern(llvm::StringRef str);

  const char *GetConstCString(const char *cstr);
  const char *GetConstCStringWithLength(const char *cstr, size_t cstr_len);
  const char *GetConstTrimmedCStringWithLength(const char *cstr,
                                               size_t cstr_len);

  /// Interns \p demangled and links it with the already interned \p mangled.
  const char *GetConstCStringAndSetMangledCounterPart(llvm::StringRef demangled,
                                                      llvm::StringRef mangled);
  const char *GetMangledCounterpart(llvm::StringRef str);

  /// Length of a string returned by any pool.
  static size_t GetConstCStringLength(const char *ccstr);

  ConstString::MemoryStats GetMemoryStats() const;

private:
  struct PoolEntry;

  PoolEntry &selectPool(uint32_t hash);
  PoolEntry &selectPool(llvm::StringRef str);

  std::unique_ptr<PoolEntry[]> m_string_pools;
};

/// A handle to a StringPool that does not own it.
class StringPoolRef {
public:
  /// Refers to the global pool.
  StringPoolRef() : m_pool(&StringPool::GetGlobal()) {}
  explicit StringPoolRef(StringPool &pool) : m_pool(&pool) {}

  const char *Intern(llvm::StringRef str) const { return m_pool->Intern(str); }

  /// Like Intern, but an empty string yields nullptr.
  const char *InternNonEmpty(llvm::StringRef str) const {
    return str.empty() ? nullptr : Intern(str);
  }

private:
  StringPool *m_pool;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_STRINGPOOL_H
