//===--- InputDependencyCollection.h - Searching for Resources --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// User-provided filters include/exclude profile instrumentation in certain
// functions.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_BASIC_INPUTDEPENDENCYCOLLECTION_H
#define LLVM_CLANG_BASIC_INPUTDEPENDENCYCOLLECTION_H

#include "clang/Basic/FileEntry.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Regex.h"
#include <string>
#include <vector>

namespace clang {

class FileManager;

/// How to handle various forms of input dependency root patterns for search
/// purposes.
enum class RootPatternScanType {
  None = 0b00,
  Directory = 0b01,
  RecursiveDirectory = 0b10,
  DirectoryAndRecursiveDirectory = 0b11
};

struct PatternFilter {
  std::string Input;
  std::string SearchRoot;
  std::string PatternRoot;
  std::string Pattern;
  llvm::Regex PatternRegex;
  RootPatternScanType RootHandling;
  bool Exported;

  PatternFilter(std::string Input);

  bool InputDependencyCheck(StringRef Input, StringRef EmbedDir,
                            StringRef Filename, FileEntryRef Fileref) const;
  bool SimpleInputDependencyCheck(StringRef Filename) const;
};

class InputDependencyCollection {
private:
  std::vector<PatternFilter> PatternFilters;

  static PatternFilter ComputeFilter(std::string Pattern, bool Exported);

public:
  InputDependencyCollection() = default;

  PatternFilter &Add(std::string Pattern, bool IsAngled, bool Exported,
                     FileManager &FM,
                     const std::vector<std::string> &SearchEntries,
                     OptionalFileEntryRef LookupFrom);

  bool InputDependencyCheck(StringRef Input, StringRef EmbedDir,
                            StringRef Filename,
                            FileEntryRef MaybeFileref) const;
  bool SimpleInputDependencyCheck(StringRef Filename) const;
};
} // namespace clang

#endif
