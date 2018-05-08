//===--- ClangTidyProfiling.cpp - clang-tidy --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangTidyProfiling.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "clang-tidy-profiling"

namespace clang {
namespace tidy {

void ClangTidyProfiling::preprocess() {
  // Convert from a insertion-friendly map to sort-friendly vector.
  Timers.clear();
  Timers.reserve(Records.size());
  for (const auto &P : Records) {
    Timers.emplace_back(P.getValue(), P.getKey());
    Total += P.getValue();
  }
  assert(Timers.size() == Records.size() && "Size mismatch after processing");

  // We want the measurements to be sorted by decreasing time spent.
  llvm::sort(Timers.begin(), Timers.end());
}

void ClangTidyProfiling::printProfileData(llvm::raw_ostream &OS) const {
  std::string Line = "===" + std::string(73, '-') + "===\n";
  OS << Line;

  if (Total.getUserTime())
    OS << "   ---User Time---";
  if (Total.getSystemTime())
    OS << "   --System Time--";
  if (Total.getProcessTime())
    OS << "   --User+System--";
  OS << "   ---Wall Time---";
  if (Total.getMemUsed())
    OS << "  ---Mem---";
  OS << "  --- Name ---\n";

  // Loop through all of the timing data, printing it out.
  for (auto I = Timers.rbegin(), E = Timers.rend(); I != E; ++I) {
    I->first.print(Total, OS);
    OS << I->second << '\n';
  }

  Total.print(Total, OS);
  OS << "Total\n";
  OS << Line << "\n";
  OS.flush();
}

ClangTidyProfiling::~ClangTidyProfiling() {
  preprocess();
  printProfileData(llvm::errs());
}

} // namespace tidy
} // namespace clang
