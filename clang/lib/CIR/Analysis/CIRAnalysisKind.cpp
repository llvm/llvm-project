//===--- CIRAnalysisKind.cpp - CIR Analysis Pass Kinds -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Analysis/CIRAnalysisKind.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace cir {

CIRAnalysisKind parseCIRAnalysisKind(llvm::StringRef name) {
 auto parseResult =  llvm::StringSwitch<CIRAnalysisKind>(name)
      .Case("fallthrough", CIRAnalysisKind::FallThrough)
      .Case("fall-through", CIRAnalysisKind::FallThrough)
      .Default(CIRAnalysisKind::Unrecognized);

 return parseResult;
}


CIRAnalysisSet parseCIRAnalysisList(
    const std::vector<std::string> &analysisList,
    llvm::SmallVectorImpl<std::string> *invalidNames) {
  CIRAnalysisSet result;

  for (const std::string &item : analysisList) {
    llvm::StringRef remaining = item;
    CIRAnalysisKind parseKind = parseCIRAnalysisKind(remaining);
    if (parseKind == CIRAnalysisKind::Unrecognized) {
      llvm::errs() << "Unrecognized CIR analysis option: "  << remaining << "\n";
      continue;
    }
    result.enable(parseKind);
  }

  return result;
}

void CIRAnalysisSet::print(llvm::raw_ostream &OS) const {
  if (empty()) {
    OS << "none";
    return;
  }

  bool first = true;
  auto printIfEnabled = [&](CIRAnalysisKind kind, llvm::StringRef name) {
    if (has(kind)) {
      if (!first)
        OS << ", ";
      OS << name;
      first = false;
    }
  };

  printIfEnabled(CIRAnalysisKind::FallThrough, "fallthrough");
  printIfEnabled(CIRAnalysisKind::UnreachableCode, "unreachable-code");
  printIfEnabled(CIRAnalysisKind::NullCheck, "null-check");
  printIfEnabled(CIRAnalysisKind::UninitializedVar, "uninitialized-var");
}

} // namespace cir
