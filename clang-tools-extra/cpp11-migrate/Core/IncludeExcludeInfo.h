//===-- Core/IncludeExcludeInfo.h - IncludeExclude class def'n --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the definition for the IncludeExcludeInfo class
/// to handle the include and exclude command line options.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_INCLUDEEXCLUDEINFO_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_INCLUDEEXCLUDEINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/system_error.h"
#include <vector>

/// \brief Class encapsulating the handling of include and exclude paths
/// provided by the user through command line options.
class IncludeExcludeInfo {
public:
  /// \brief Read and parse a comma-seperated lists of paths from
  /// \a IncludeString and \a ExcludeString.
  ///
  /// Returns error_code::success() on successful parse of the strings or
  /// an error_code indicating the encountered error.
  llvm::error_code readListFromString(llvm::StringRef IncludeString,
                                      llvm::StringRef ExcludeString);

  /// \brief Read and parse the lists of paths from \a IncludeListFile
  /// and \a ExcludeListFile. Each file should contain one path per line.
  ///
  /// Returns error_code::success() on successful read and parse of both files
  /// or an error_code indicating the encountered error.
  llvm::error_code readListFromFile(llvm::StringRef IncludeListFile,
                                    llvm::StringRef ExcludeListFile);

  /// \brief Determine if the given path is in the list of include paths but
  /// not in the list of exclude paths.
  bool isFileIncluded(llvm::StringRef FilePath);

private:
  std::vector<std::string> IncludeList;
  std::vector<std::string> ExcludeList;
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_INCLUDEEXCLUDEINFO_H
