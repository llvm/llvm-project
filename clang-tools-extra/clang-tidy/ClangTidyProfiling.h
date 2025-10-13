//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYPROFILING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYPROFILING_H

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/Timer.h"
#include <optional>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace clang::tidy {

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
  std::optional<StorageParams> Storage;

  void printUserFriendlyTable(llvm::raw_ostream &OS, llvm::TimerGroup &TG);
  void printAsJSON(llvm::raw_ostream &OS, llvm::TimerGroup &TG);
  void storeProfileData(llvm::TimerGroup &TG);

public:
  llvm::StringMap<llvm::TimeRecord> Records;

  ClangTidyProfiling() = default;

  ClangTidyProfiling(std::optional<StorageParams> Storage);

  ~ClangTidyProfiling();
};

} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYPROFILING_H
