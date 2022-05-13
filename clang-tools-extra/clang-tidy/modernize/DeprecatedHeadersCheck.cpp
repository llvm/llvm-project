//===--- DeprecatedHeadersCheck.cpp - clang-tidy---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeprecatedHeadersCheck.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <vector>

namespace clang {
namespace tidy {
namespace modernize {
namespace detail {
bool operator<(const IncludeMarker &LHS, const IncludeMarker &RHS) {
  return LHS.DecomposedDiagLoc < RHS.DecomposedDiagLoc;
}
bool operator<(const IncludeMarker &LHS,
               const std::pair<FileID, unsigned> &RHS) {
  return LHS.DecomposedDiagLoc < RHS;
}
bool operator<(const std::pair<FileID, unsigned> &LHS,
               const IncludeMarker &RHS) {
  return LHS < RHS.DecomposedDiagLoc;
}

class IncludeModernizePPCallbacks : public PPCallbacks {
public:
  explicit IncludeModernizePPCallbacks(DeprecatedHeadersCheck &Check,
                                       LangOptions LangOpts,
                                       const SourceManager &SM);

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          Optional<FileEntryRef> File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;

private:
  DeprecatedHeadersCheck &Check;
  LangOptions LangOpts;
  llvm::StringMap<std::string> CStyledHeaderToCxx;
  llvm::StringSet<> DeleteHeaders;
  const SourceManager &SM;
};

class ExternCRefutationVisitor
    : public RecursiveASTVisitor<ExternCRefutationVisitor> {
  std::vector<IncludeMarker> &IncludesToBeProcessed;
  const SourceManager &SM;

public:
  ExternCRefutationVisitor(std::vector<IncludeMarker> &IncludesToBeProcessed,
                           SourceManager &SM)
      : IncludesToBeProcessed(IncludesToBeProcessed), SM(SM) {}
  bool shouldWalkTypesOfTypeLocs() const { return false; }
  bool shouldVisitLambdaBody() const { return false; }

  bool VisitLinkageSpecDecl(LinkageSpecDecl *LinkSpecDecl) const {
    if (LinkSpecDecl->getLanguage() != LinkageSpecDecl::lang_c ||
        !LinkSpecDecl->hasBraces())
      return true;

    auto ExternCBlockBegin =
        SM.getDecomposedExpansionLoc(LinkSpecDecl->getBeginLoc());
    auto ExternCBlockEnd =
        SM.getDecomposedExpansionLoc(LinkSpecDecl->getEndLoc());

    auto Begin = IncludesToBeProcessed.begin();
    auto End = IncludesToBeProcessed.end();
    auto Low = std::lower_bound(Begin, End, ExternCBlockBegin);
    auto Up = std::upper_bound(Low, End, ExternCBlockEnd);
    IncludesToBeProcessed.erase(Low, Up);
    return true;
  }
};
} // namespace detail

void DeprecatedHeadersCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(::std::make_unique<detail::IncludeModernizePPCallbacks>(
      *this, getLangOpts(), PP->getSourceManager()));
}
void DeprecatedHeadersCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  // Even though the checker operates on a "preprocessor" level, we still need
  // to act on a "TranslationUnit" to acquire the AST where we can walk each
  // Decl and look for `extern "C"` blocks where we will suppress the report we
  // collected during the preprocessing phase.
  // The `onStartOfTranslationUnit()` won't suffice, since we need some handle
  // to the `ASTContext`.
  Finder->addMatcher(ast_matchers::translationUnitDecl().bind("TU"), this);
}

void DeprecatedHeadersCheck::onEndOfTranslationUnit() {
  IncludesToBeProcessed.clear();
}

void DeprecatedHeadersCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  SourceManager &SM = Result.Context->getSourceManager();
  using detail::IncludeMarker;

  llvm::sort(IncludesToBeProcessed);

  // Suppress includes wrapped by `extern "C" { ... }` blocks.
  detail::ExternCRefutationVisitor Visitor(IncludesToBeProcessed, SM);
  Visitor.TraverseAST(*Result.Context);

  // Emit all the remaining reports.
  for (const IncludeMarker &Entry : IncludesToBeProcessed) {
    SourceLocation DiagLoc = SM.getComposedLoc(Entry.DecomposedDiagLoc.first,
                                               Entry.DecomposedDiagLoc.second);
    if (Entry.Replacement.empty()) {
      diag(DiagLoc, "including '%0' has no effect in C++; consider removing it")
          << Entry.FileName << FixItHint::CreateRemoval(Entry.ReplacementRange);
    } else {
      diag(DiagLoc, "inclusion of deprecated C++ header "
                    "'%0'; consider using '%1' instead")
          << Entry.FileName << Entry.Replacement
          << FixItHint::CreateReplacement(
                 Entry.ReplacementRange,
                 (llvm::Twine("<") + Entry.Replacement + ">").str());
    }
  }
}

detail::IncludeModernizePPCallbacks::IncludeModernizePPCallbacks(
    DeprecatedHeadersCheck &Check, LangOptions LangOpts,
    const SourceManager &SM)
    : Check(Check), LangOpts(LangOpts), SM(SM) {
  for (const auto &KeyValue :
       std::vector<std::pair<llvm::StringRef, std::string>>(
           {{"assert.h", "cassert"},
            {"complex.h", "complex"},
            {"ctype.h", "cctype"},
            {"errno.h", "cerrno"},
            {"float.h", "cfloat"},
            {"limits.h", "climits"},
            {"locale.h", "clocale"},
            {"math.h", "cmath"},
            {"setjmp.h", "csetjmp"},
            {"signal.h", "csignal"},
            {"stdarg.h", "cstdarg"},
            {"stddef.h", "cstddef"},
            {"stdio.h", "cstdio"},
            {"stdlib.h", "cstdlib"},
            {"string.h", "cstring"},
            {"time.h", "ctime"},
            {"wchar.h", "cwchar"},
            {"wctype.h", "cwctype"}})) {
    CStyledHeaderToCxx.insert(KeyValue);
  }
  // Add C++ 11 headers.
  if (LangOpts.CPlusPlus11) {
    for (const auto &KeyValue :
         std::vector<std::pair<llvm::StringRef, std::string>>(
             {{"fenv.h", "cfenv"},
              {"stdint.h", "cstdint"},
              {"inttypes.h", "cinttypes"},
              {"tgmath.h", "ctgmath"},
              {"uchar.h", "cuchar"}})) {
      CStyledHeaderToCxx.insert(KeyValue);
    }
  }
  for (const auto &Key :
       std::vector<std::string>({"stdalign.h", "stdbool.h", "iso646.h"})) {
    DeleteHeaders.insert(Key);
  }
}

void detail::IncludeModernizePPCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, Optional<FileEntryRef> File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  // FIXME: Take care of library symbols from the global namespace.
  //
  // Reasonable options for the check:
  //
  // 1. Insert std prefix for every such symbol occurrence.
  // 2. Insert `using namespace std;` to the beginning of TU.
  // 3. Do nothing and let the user deal with the migration himself.
  std::pair<FileID, unsigned> DiagLoc =
      SM.getDecomposedExpansionLoc(FilenameRange.getBegin());
  if (CStyledHeaderToCxx.count(FileName) != 0) {
    Check.IncludesToBeProcessed.push_back(
        IncludeMarker{CStyledHeaderToCxx[FileName], FileName,
                      FilenameRange.getAsRange(), DiagLoc});
  } else if (DeleteHeaders.count(FileName) != 0) {
    Check.IncludesToBeProcessed.push_back(
        IncludeMarker{std::string{}, FileName,
                      SourceRange{HashLoc, FilenameRange.getEnd()}, DiagLoc});
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
