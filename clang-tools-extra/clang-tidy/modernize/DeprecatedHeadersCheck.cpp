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
#include "llvm/ADT/StringSet.h"

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
                          const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;

private:
  ClangTidyCheck &Check;
  LangOptions LangOpts;
  llvm::StringMap<std::string> CStyledHeaderToCxx;
  llvm::StringSet<> DeleteHeaders;
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
    : Check(Check), LangOpts(LangOpts) {
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

void IncludeModernizePPCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  // FIXME: Take care of library symbols from the global namespace.
  //
  // Reasonable options for the check:
  //
  // 1. Insert std prefix for every such symbol occurrence.
  // 2. Insert `using namespace std;` to the beginning of TU.
  // 3. Do nothing and let the user deal with the migration himself.
  if (CStyledHeaderToCxx.count(FileName) != 0) {
    std::string Replacement =
        (llvm::Twine("<") + CStyledHeaderToCxx[FileName] + ">").str();
    Check.diag(FilenameRange.getBegin(), "inclusion of deprecated C++ header "
                                         "'%0'; consider using '%1' instead")
        << FileName << CStyledHeaderToCxx[FileName]
        << FixItHint::CreateReplacement(FilenameRange.getAsRange(),
                                        Replacement);
  } else if (DeleteHeaders.count(FileName) != 0) {
    Check.diag(FilenameRange.getBegin(),
               "including '%0' has no effect in C++; consider removing it")
        << FileName << FixItHint::CreateRemoval(
                           SourceRange(HashLoc, FilenameRange.getEnd()));
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
