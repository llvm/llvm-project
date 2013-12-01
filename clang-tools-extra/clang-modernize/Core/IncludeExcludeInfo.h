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

#ifndef CLANG_MODERNIZE_INCLUDEEXCLUDEINFO_H
#define CLANG_MODERNIZE_INCLUDEEXCLUDEINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/system_error.h"
#include <vector>

/// \brief Class encapsulating the handling of include and exclude paths
/// provided by the user through command line options.
class IncludeExcludeInfo {
public:
  /// \brief Read and parse a comma-separated lists of paths from
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
  ///
  /// \a FilePath shouldn't contain relative operators i.e. ".." or "." since
  /// Path comes from the include/exclude list of paths in which relative
  /// operators were removed.
  bool isFileIncluded(llvm::StringRef FilePath) const;

  /// \brief Determine if a file path was explicitly excluded.
  bool isFileExplicitlyExcluded(llvm::StringRef FilePath) const;

  /// \brief Determine if a list of include paths was provided.
  bool isIncludeListEmpty() const { return IncludeList.empty(); }

private:
  std::vector<std::string> IncludeList;
  std::vector<std::string> ExcludeList;
};

#endif // CLANG_MODERNIZE_INCLUDEEXCLUDEINFO_H
