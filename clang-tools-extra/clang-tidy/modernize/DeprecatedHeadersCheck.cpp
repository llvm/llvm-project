//===----------------------------------------------------------------------===//
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

#include <vector>

using IncludeMarker =
    clang::tidy::modernize::DeprecatedHeadersCheck::IncludeMarker;
namespace clang::tidy::modernize {
namespace {

class IncludeModernizePPCallbacks : public PPCallbacks {
public:
  explicit IncludeModernizePPCallbacks(
      std::vector<IncludeMarker> &IncludesToBeProcessed,
      const LangOptions &LangOpts, const SourceManager &SM,
      bool CheckHeaderFile);

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *SuggestedModule,
                          bool ModuleImported,
                          SrcMgr::CharacteristicKind FileType) override;

private:
  std::vector<IncludeMarker> &IncludesToBeProcessed;
  llvm::StringMap<StringRef> CStyledHeaderToCxx;
  llvm::StringSet<> DeleteHeaders;
  const SourceManager &SM;
  bool CheckHeaderFile;
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
    if (LinkSpecDecl->getLanguage() != LinkageSpecLanguageIDs::C ||
        !LinkSpecDecl->hasBraces())
      return true;

    auto ExternCBlockBegin = LinkSpecDecl->getBeginLoc();
    auto ExternCBlockEnd = LinkSpecDecl->getEndLoc();
    auto IsWrapped = [=, &SM = SM](const IncludeMarker &Marker) -> bool {
      return SM.isBeforeInTranslationUnit(ExternCBlockBegin, Marker.DiagLoc) &&
             SM.isBeforeInTranslationUnit(Marker.DiagLoc, ExternCBlockEnd);
    };

    llvm::erase_if(IncludesToBeProcessed, IsWrapped);
    return true;
  }
};
} // namespace

DeprecatedHeadersCheck::DeprecatedHeadersCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CheckHeaderFile(Options.get("CheckHeaderFile", false)) {}

void DeprecatedHeadersCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckHeaderFile", CheckHeaderFile);
}

void DeprecatedHeadersCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(std::make_unique<IncludeModernizePPCallbacks>(
      IncludesToBeProcessed, getLangOpts(), PP->getSourceManager(),
      CheckHeaderFile));
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

  // Suppress includes wrapped by `extern "C" { ... }` blocks.
  ExternCRefutationVisitor Visitor(IncludesToBeProcessed, SM);
  Visitor.TraverseAST(*Result.Context);

  // Emit all the remaining reports.
  for (const IncludeMarker &Marker : IncludesToBeProcessed) {
    if (Marker.Replacement.empty()) {
      diag(Marker.DiagLoc,
           "including '%0' has no effect in C++; consider removing it")
          << Marker.FileName
          << FixItHint::CreateRemoval(Marker.ReplacementRange);
    } else {
      diag(Marker.DiagLoc, "inclusion of deprecated C++ header "
                           "'%0'; consider using '%1' instead")
          << Marker.FileName << Marker.Replacement
          << FixItHint::CreateReplacement(
                 Marker.ReplacementRange,
                 (llvm::Twine("<") + Marker.Replacement + ">").str());
    }
  }
}

IncludeModernizePPCallbacks::IncludeModernizePPCallbacks(
    std::vector<IncludeMarker> &IncludesToBeProcessed,
    const LangOptions &LangOpts, const SourceManager &SM, bool CheckHeaderFile)
    : IncludesToBeProcessed(IncludesToBeProcessed), SM(SM),
      CheckHeaderFile(CheckHeaderFile) {

  static constexpr std::pair<StringRef, StringRef> CXX98Headers[] = {
      {"assert.h", "cassert"}, {"complex.h", "complex"},
      {"ctype.h", "cctype"},   {"errno.h", "cerrno"},
      {"float.h", "cfloat"},   {"limits.h", "climits"},
      {"locale.h", "clocale"}, {"math.h", "cmath"},
      {"setjmp.h", "csetjmp"}, {"signal.h", "csignal"},
      {"stdarg.h", "cstdarg"}, {"stddef.h", "cstddef"},
      {"stdio.h", "cstdio"},   {"stdlib.h", "cstdlib"},
      {"string.h", "cstring"}, {"time.h", "ctime"},
      {"wchar.h", "cwchar"},   {"wctype.h", "cwctype"},
  };
  CStyledHeaderToCxx.insert(std::begin(CXX98Headers), std::end(CXX98Headers));

  static constexpr std::pair<StringRef, StringRef> CXX11Headers[] = {
      {"fenv.h", "cfenv"},         {"stdint.h", "cstdint"},
      {"inttypes.h", "cinttypes"}, {"tgmath.h", "ctgmath"},
      {"uchar.h", "cuchar"},
  };
  if (LangOpts.CPlusPlus11)
    CStyledHeaderToCxx.insert(std::begin(CXX11Headers), std::end(CXX11Headers));

  static constexpr StringRef HeadersToDelete[] = {"stdalign.h", "stdbool.h",
                                                  "iso646.h"};
  DeleteHeaders.insert_range(HeadersToDelete);
}

void IncludeModernizePPCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, OptionalFileEntryRef File,
    StringRef SearchPath, StringRef RelativePath, const Module *SuggestedModule,
    bool ModuleImported, SrcMgr::CharacteristicKind FileType) {

  // If we don't want to warn for non-main file reports and this is one, skip
  // it.
  if (!CheckHeaderFile && !SM.isInMainFile(HashLoc))
    return;

  // Ignore system headers.
  if (SM.isInSystemHeader(HashLoc))
    return;

  // FIXME: Take care of library symbols from the global namespace.
  //
  // Reasonable options for the check:
  //
  // 1. Insert std prefix for every such symbol occurrence.
  // 2. Insert `using namespace std;` to the beginning of TU.
  // 3. Do nothing and let the user deal with the migration himself.
  SourceLocation DiagLoc = FilenameRange.getBegin();
  if (auto It = CStyledHeaderToCxx.find(FileName);
      It != CStyledHeaderToCxx.end()) {
    IncludesToBeProcessed.emplace_back(IncludeMarker{
        It->second, FileName, FilenameRange.getAsRange(), DiagLoc});
  } else if (DeleteHeaders.contains(FileName)) {
    IncludesToBeProcessed.emplace_back(
        // NOLINTNEXTLINE(modernize-use-emplace) - false-positive
        IncludeMarker{StringRef{}, FileName,
                      SourceRange{HashLoc, FilenameRange.getEnd()}, DiagLoc});
  }
}

} // namespace clang::tidy::modernize
