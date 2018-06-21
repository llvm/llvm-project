//===--- ClangTidyProfiling.h - clang-tidy ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYPROFILING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYPROFILING_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace tidy {

class ClangTidyProfiling {
public:
  struct StorageParams {
    llvm::sys::TimePoint<> Timestamp;
    std::string SourceFilename;
    std::string StoreFilename;

    StorageParams() = default;

    StorageParams(llvm::StringRef ProfilePrefix, llvm::StringRef SourceFile);
  };

private:
  llvm::Optional<llvm::TimerGroup> TG;

  llvm::Optional<StorageParams> Storage;

  void printUserFriendlyTable(llvm::raw_ostream &OS);
  void printAsJSON(llvm::raw_ostream &OS);

  void storeProfileData();

public:
  llvm::StringMap<llvm::TimeRecord> Records;

  ClangTidyProfiling() = default;

  ClangTidyProfiling(llvm::Optional<StorageParams> Storage);

  ~ClangTidyProfiling();
};

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYPROFILING_H
