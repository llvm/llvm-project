//===-- ConfigTesting.h - Helpers for configuration tests -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_CONFIGTESTING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_CONFIGTESTING_H

#include "Protocol.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include <functional>

namespace clang {
namespace clangd {
namespace config {

// Provides a DiagnosticsCallback that records diganostics.
// Unlike just pushing them into a vector, underlying storage need not survive.
struct CapturedDiags {
  std::function<void(const llvm::SMDiagnostic &)> callback() {
    return [this](const llvm::SMDiagnostic &D) {
      if (Files.empty() || Files.back() != D.getFilename())
        Files.push_back(D.getFilename().str());

      if (D.getKind() > llvm::SourceMgr::DK_Warning)
        return;

      Diagnostics.emplace_back();
      Diag &Out = Diagnostics.back();
      Out.Message = D.getMessage().str();
      Out.Kind = D.getKind();
      Out.Pos.line = D.getLineNo() - 1;
      Out.Pos.character = D.getColumnNo(); // Zero-based - bug in SourceMgr?
      if (!D.getRanges().empty()) {
        const auto &R = D.getRanges().front();
        Out.Rng.emplace();
        Out.Rng->start.line = Out.Rng->end.line = Out.Pos.line;
        Out.Rng->start.character = R.first;
        Out.Rng->end.character = R.second;
      }
    };
  }
  struct Diag {
    std::string Message;
    llvm::SourceMgr::DiagKind Kind;
    Position Pos;
    llvm::Optional<Range> Rng;

    friend void PrintTo(const Diag &D, std::ostream *OS) {
      *OS << (D.Kind == llvm::SourceMgr::DK_Error ? "error: " : "warning: ")
          << D.Message << "@" << llvm::to_string(D.Pos);
    }
  };
  std::vector<Diag> Diagnostics;  // Warning or higher.
  std::vector<std::string> Files; // Filename from diagnostics including notes.

  void clear() {
    Diagnostics.clear();
    Files.clear();
  }
};

MATCHER_P(DiagMessage, M, "") { return arg.Message == M; }
MATCHER_P(DiagKind, K, "") { return arg.Kind == K; }
MATCHER_P(DiagPos, P, "") { return arg.Pos == P; }
MATCHER_P(DiagRange, R, "") { return arg.Rng == R; }

inline Position toPosition(llvm::SMLoc L, const llvm::SourceMgr &SM) {
  auto LineCol = SM.getLineAndColumn(L);
  Position P;
  P.line = LineCol.first - 1;
  P.character = LineCol.second - 1;
  return P;
}

inline Range toRange(llvm::SMRange R, const llvm::SourceMgr &SM) {
  return Range{toPosition(R.Start, SM), toPosition(R.End, SM)};
}

} // namespace config
} // namespace clangd
} // namespace clang

#endif
