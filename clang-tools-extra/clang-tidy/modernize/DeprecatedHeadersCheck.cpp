//===--- DeprecatedHeadersCheck.cpp - clang-tidy---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DeprecatedHeadersCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/StringMap.h"

#include <vector>

namespace clang {
namespace tidy {
namespace modernize {

namespace {
class IncludeModernizePPCallbacks : public PPCallbacks {
public:
  explicit IncludeModernizePPCallbacks(ClangTidyCheck &Check,
                                       LangOptions LangOpts);

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported) override;

private:
  ClangTidyCheck &Check;
  LangOptions LangOpts;
  llvm::StringMap<std::string> CStyledHeaderToCxx;
};
} // namespace

void DeprecatedHeadersCheck::registerPPCallbacks(CompilerInstance &Compiler) {
  if (this->getLangOpts().CPlusPlus) {
    Compiler.getPreprocessor().addPPCallbacks(
        ::llvm::make_unique<IncludeModernizePPCallbacks>(*this,
                                                         this->getLangOpts()));
  }
}

IncludeModernizePPCallbacks::IncludeModernizePPCallbacks(ClangTidyCheck &Check,
                                                         LangOptions LangOpts)
    : Check(Check), LangOpts(LangOpts),
      CStyledHeaderToCxx({{"assert.h", "cassert"},
                          {"complex.h", "ccomplex"},
                          {"ctype.h", "cctype"},
                          {"errno.h", "cerrno"},
                          {"float.h", "cfloat"},
                          {"inttypes.h", "cinttypes"},
                          {"iso646.h", "ciso646"},
                          {"limits.h", "climits"},
                          {"locale.h", "clocale"},
                          {"math.h", "cmath"},
                          {"setjmp.h", "csetjmp"},
                          {"signal.h", "csignal"},
                          {"stdarg.h", "cstdarg"},
                          {"stddef.h", "cstddef"},
                          {"stdint.h", "cstdint"},
                          {"stdio.h", "cstdio"},
                          {"stdlib.h", "cstdlib"},
                          {"string.h", "cstring"},
                          {"time.h", "ctime"},
                          {"wchar.h", "cwchar"},
                          {"wctype.h", "cwctype"}}) {
  // Add C++ 11 headers.
  if (LangOpts.CPlusPlus11) {
    for (const auto &it : std::vector<std::pair<std::string, std::string>>(
             {{"fenv.h", "cfenv"},
              {"stdalign.h", "cstdalign"},
              {"stdbool.h", "cstdbool"},
              {"tgmath.h", "ctgmath"},
              {"uchar.h", "cuchar"}})) {
      CStyledHeaderToCxx.insert(it);
    }
  }
}

void IncludeModernizePPCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported) {
  // FIXME: Take care of library symbols from the global namespace.
  //
  // Reasonable options for the check:
  //
  // 1. Insert std prefix for every such symbol occurance.
  // 2. Insert `using namespace std;` to the beginning of TU.
  // 3. Do nothing and let the user deal with the migration himself.
  if (CStyledHeaderToCxx.count(FileName) != 0) {
    std::string Replacement =
        (llvm::Twine("<") + CStyledHeaderToCxx[FileName] + ">").str();
    Check.diag(FilenameRange.getBegin(),
               "inclusion of deprecated C++ header '%0'; consider using '%1' instead")
        << FileName << CStyledHeaderToCxx[FileName]
        << FixItHint::CreateReplacement(FilenameRange.getAsRange(),
                                        Replacement);
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
