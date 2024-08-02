//===-- RealpathPrefixes.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_REALPATHPREFIXES_H
#define LLDB_CORE_REALPATHPREFIXES_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <optional>
#include <string>
#include <vector>

namespace lldb_private {
class FileSpec;
class FileSpecList;
class Target;
} // namespace lldb_private

namespace lldb_private {

class RealpathPrefixes {
public:
  // Prefixes are obtained from FileSpecList, through FileSpec::GetPath(), which
  // ensures that the paths are normalized. For example:
  // "./foo/.." -> ""
  // "./foo/../bar" -> "bar"
  explicit RealpathPrefixes(const FileSpecList &file_spec_list);

  // Sets an optional filesystem to use for realpath'ing. If not set, the real
  // filesystem will be used.
  void SetFileSystem(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs);

  // Sets an optional Target instance to gather statistics.
  void SetTarget(Target *target) { m_target = target; }
  Target *GetTarget() const { return m_target; }

  std::optional<FileSpec> ResolveSymlinks(const FileSpec &file_spec) const;

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
  Target *m_target;
};

} // namespace lldb_private

#endif // LLDB_CORE_REALPATHPREFIXES_H
