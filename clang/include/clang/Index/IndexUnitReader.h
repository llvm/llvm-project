//===--- IndexUnitReader.h - Index unit deserialization -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_INDEXUNITREADER_H
#define LLVM_CLANG_INDEX_INDEXUNITREADER_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/PathRemapper.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Chrono.h"

namespace clang {
namespace index {

class IndexUnitReader {
public:
  enum class DependencyKind {
    Unit,
    Record,
    File,
  };

  ~IndexUnitReader();

  static std::unique_ptr<IndexUnitReader>
    createWithUnitFilename(StringRef UnitFilename, StringRef StorePath,
                           const PathRemapper &Remapper, std::string &Error);
  static std::unique_ptr<IndexUnitReader>
    createWithFilePath(StringRef FilePath, const PathRemapper &Remapper,
                       std::string &Error);

  static Optional<llvm::sys::TimePoint<>>
    getModificationTimeForUnit(StringRef UnitFilename, StringRef StorePath,
                               std::string &Error);

  StringRef getProviderIdentifier() const;
  StringRef getProviderVersion() const;

  llvm::sys::TimePoint<> getModificationTime() const;
  StringRef getWorkingDirectory() const;
  StringRef getOutputFile() const;
  StringRef getSysrootPath() const;
  StringRef getMainFilePath() const;
  StringRef getModuleName() const;
  StringRef getTarget() const;
  bool hasMainFile() const;
  bool isSystemUnit() const;
  bool isModuleUnit() const;
  bool isDebugCompilation() const;

  struct DependencyInfo {
    DependencyKind Kind;
    bool IsSystem;
    StringRef UnitOrRecordName;
    StringRef FilePath;
    StringRef ModuleName;
  };
  struct IncludeInfo {
    StringRef SourcePath;
    unsigned SourceLine;
    StringRef TargetPath;
  };
  /// Unit dependencies are provided ahead of record ones, record ones
  /// ahead of the file ones.
  bool foreachDependency(llvm::function_ref<bool(const DependencyInfo &Info)> Receiver);

  bool foreachInclude(llvm::function_ref<bool(const IncludeInfo &Info)> Receiver);

private:
  IndexUnitReader(void *Impl) : Impl(Impl) {}

  void *Impl; // An IndexUnitReaderImpl.
};

} // namespace index
} // namespace clang

#endif
