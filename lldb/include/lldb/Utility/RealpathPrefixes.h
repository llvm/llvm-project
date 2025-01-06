//===-- RealpathPrefixes.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_REALPATHPREFIXES_H
#define LLDB_UTILITY_REALPATHPREFIXES_H

#include "lldb/lldb-forward.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <optional>
#include <string>
#include <vector>

namespace lldb_private {

class RealpathPrefixes {
public:
  /// \param[in] file_spec_list
  ///     Prefixes are obtained from FileSpecList, through FileSpec::GetPath(),
  ///     which ensures that the paths are normalized. For example:
  ///     "./foo/.." -> ""
  ///     "./foo/../bar" -> "bar"
  ///
  /// \param[in] fs
  ///     An optional filesystem to use for realpath'ing. If not set, the real
  ///     filesystem will be used.
  explicit RealpathPrefixes(const FileSpecList &file_spec_list,
                            llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs =
                                llvm::vfs::getRealFileSystem());

  std::optional<FileSpec> ResolveSymlinks(const FileSpec &file_spec);

  // If/when Statistics.h/cpp is moved into Utility, we can remove these
  // methods, hold a (weak) pointer to `TargetStats` and directly increment
  // on that object.
  void IncreaseSourceRealpathAttemptCount() {
    ++m_source_realpath_attempt_count;
  }
  uint32_t GetSourceRealpathAttemptCount() const {
    return m_source_realpath_attempt_count;
  }
  void IncreaseSourceRealpathCompatibleCount() {
    ++m_source_realpath_compatible_count;
  }
  uint32_t GetSourceRealpathCompatibleCount() const {
    return m_source_realpath_compatible_count;
  }

private:
  // Paths that start with one of the prefixes in this list will be realpath'ed
  // to resolve any symlinks.
  //
  // Wildcard prefixes:
  // - "" (empty string) will match all paths.
  // - "/" will match all absolute paths.
  std::vector<std::string> m_prefixes;

  // The filesystem to use for realpath'ing.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> m_fs;

  // The optional Target instance to gather statistics.
  lldb::TargetWP m_target;

  // Statistics that we temprarily hold here, to be gathered into TargetStats
  uint32_t m_source_realpath_attempt_count = 0;
  uint32_t m_source_realpath_compatible_count = 0;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_REALPATHPREFIXES_H
