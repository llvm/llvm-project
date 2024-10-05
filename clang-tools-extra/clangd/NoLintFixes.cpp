//===--- NoLintFixes.cpp -------------------------------------------*-
// C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoLintFixes.h"
#include "../clang-tidy/ClangTidyCheck.h"
#include "../clang-tidy/ClangTidyDiagnosticConsumer.h"
#include "../clang-tidy/ClangTidyModule.h"
#include "AST.h"
#include "Diagnostics.h"
#include "FeatureModule.h"
#include "SourceCode.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/Lexer.h"
#include "clang/Serialization/ASTWriter.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <cctype>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

std::vector<Fix>
noLintFixes(const clang::tidy::ClangTidyContext &CTContext,
                     const clang::Diagnostic &Info, const Diag &Diag) {
  auto RuleName = CTContext.getCheckName(Diag.ID);
  if (
      // If this isn't a clang-tidy diag
      RuleName.empty() ||
      // NOLINT does not work on Serverity Error or above
      Diag.Severity >= DiagnosticsEngine::Error ||
      // No point adding extra fixes if the Diag is for a different file
      !Diag.InsideMainFile) {
    return {};
  }

  auto &SrcMgr = Info.getSourceManager();
  auto &DiagLoc = Info.getLocation();

  auto F = Fix{};
  F.Message = llvm::formatv("ignore [{0}] for this line", RuleName);
  auto &E = F.Edits.emplace_back();

  auto File = SrcMgr.getFileID(DiagLoc);
  auto CodeTilDiag = toSourceCode(
      SrcMgr, SourceRange(SrcMgr.getLocForStartOfFile(File), DiagLoc));

  auto StartCurLine = CodeTilDiag.find_last_of('\n') + 1;
  auto CurLine = CodeTilDiag.substr(StartCurLine);
  auto Indent = CurLine.take_while([](char C) { return std::isspace(C); });

  auto ExistingNoLintNextLineInsertPos = std::optional<int>();
  if (StartCurLine > 0) {
    auto StartPrevLine = CodeTilDiag.find_last_of('\n', StartCurLine - 1) + 1;
    auto PrevLine =
        CodeTilDiag.substr(StartPrevLine, StartCurLine - StartPrevLine - 1);
    auto NLPos = PrevLine.find("NOLINTNEXTLINE(");
    if (NLPos != StringRef::npos) {
      ExistingNoLintNextLineInsertPos =
          std::make_optional(PrevLine.find(")", NLPos));
    }
  }

  auto InsertPos = sourceLocToPosition(SrcMgr, DiagLoc);
  if (ExistingNoLintNextLineInsertPos) {
    E.newText = llvm::formatv(", {0}", RuleName);
    InsertPos.line -= 1;
    InsertPos.character = *ExistingNoLintNextLineInsertPos;
  } else {
    E.newText = llvm::formatv("{0}// NOLINTNEXTLINE({1})\n", Indent, RuleName);
    InsertPos.character = 0;
  }
  E.range = {InsertPos, InsertPos};

  return {F};
}

const auto Regex = std::regex(NoLintFixMsgRegexStr);
bool isNoLintFixes(const Fix &F) {
  return std::regex_match(F.Message, Regex);
}

} // namespace clangd
} // namespace clang
