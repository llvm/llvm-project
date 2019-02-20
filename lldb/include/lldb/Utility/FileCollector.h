//===-- FileCollector.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_FILE_COLLECTOR_H
#define LLDB_UTILITY_FILE_COLLECTOR_H

#include "lldb/Utility/FileSpec.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <mutex>

namespace lldb_private {

/// Collects files into a directory and generates a mapping that can be used by
/// the VFS.
class FileCollector {
public:
  FileCollector(const FileSpec &root);

  void AddFile(const llvm::Twine &file);
  void AddFile(const FileSpec &file) { return AddFile(file.GetPath()); }

  /// Write the yaml mapping (for the VFS) to the given file.
  std::error_code WriteMapping(const FileSpec &mapping_file);

  /// Copy the files into the root directory.
  ///
  /// When stop_on_error is true (the default) we abort as soon as one file
  /// cannot be copied. This is relatively common, for example when a file was
  /// removed after it was added to the mapping.
  std::error_code CopyFiles(bool stop_on_error = true);

protected:
  void AddFileImpl(llvm::StringRef src_path);

  bool MarkAsSeen(llvm::StringRef path) { return m_seen.insert(path).second; }

  bool GetRealPath(llvm::StringRef src_path,
                   llvm::SmallVectorImpl<char> &result);

  void AddFileToMapping(llvm::StringRef virtual_path,
                        llvm::StringRef real_path) {
    m_vfs_writer.addFileMapping(virtual_path, real_path);
  }

  /// Synchronizes adding files.
  std::mutex m_mutex;

  /// The root directory where files are copied.
  FileSpec m_root;

  /// Tracks already seen files so they can be skipped.
  llvm::StringSet<> m_seen;

  /// The yaml mapping writer.
  llvm::vfs::YAMLVFSWriter m_vfs_writer;

  /// Caches real_path calls when resolving symlinks.
  llvm::StringMap<std::string> m_symlink_map;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_FILE_COLLECTOR_H
