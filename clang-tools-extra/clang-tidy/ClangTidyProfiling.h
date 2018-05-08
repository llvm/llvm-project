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

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>
#include <vector>

namespace clang {
namespace tidy {

class ClangTidyProfiling {
  // Time is first to allow for sorting by it.
  std::vector<std::pair<llvm::TimeRecord, llvm::StringRef>> Timers;
  llvm::TimeRecord Total;

  void preprocess();

  void printProfileData(llvm::raw_ostream &OS) const;

public:
  llvm::StringMap<llvm::TimeRecord> Records;

  ~ClangTidyProfiling();
};

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYPROFILING_H
