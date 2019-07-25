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

#include "llvm/Support/FileCollector.h"

namespace lldb_private {

/// Collects files into a directory and generates a mapping that can be used by
/// the VFS.
class FileCollector : public llvm::FileCollector {
public:
  FileCollector(const FileSpec &root, const FileSpec &overlay) :
    llvm::FileCollector(root.GetPath(), overlay.GetPath()) {}

  using llvm::FileCollector::addFile;

  void addFile(const FileSpec &file) {
    std::string path = file.GetPath();
    llvm::FileCollector::addFile(path);
  }

  /// Write the yaml mapping (for the VFS) to the given file.
  std::error_code writeMapping(const FileSpec &mapping_file) {
    std::string path = mapping_file.GetPath();
    return llvm::FileCollector::writeMapping(path);
  }
};

} // namespace lldb_private

#endif // LLDB_UTILITY_FILE_COLLECTOR_H
