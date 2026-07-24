//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeaderGuardCheck.h"
#include "../utils/LexerUtils.h"
#include "../utils/OptionsUtils.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Path.h"

namespace clang::tidy::misc {

/// Canonicalize a path by removing ./ and ../ components, matching the
/// format of the ``Filename`` passed to the ``HeaderGuardCheck`` virtuals.
static std::string cleanPath(StringRef Path) {
  SmallString<256> Result = Path;
  llvm::sys::path::remove_dots(Result, true);
  return std::string(Result);
}

HeaderGuardCheck::HeaderGuardCheck(StringRef Name, ClangTidyContext *Context)
    : clang::tidy::utils::HeaderGuardCheck(Name, Context),
      AllowPragmaOnce(Options.get("AllowPragmaOnce", false)),
      HeaderDirs(utils::options::parseStringList(
          Options.get("HeaderDirs", "include"))),
      EndifComment(Options.get("EndifComment", false)),
      Prefix(Options.get("Prefix", "")) {}

std::string HeaderGuardCheck::getHeaderGuard(StringRef Filename,
                                             StringRef /*OldGuard*/) {
  // When running under Windows, need to convert the path separators from
  // `\` to `/`.
  std::string AbsPath =
      llvm::sys::path::convert_to_slash(tooling::getAbsolutePath(Filename));

  // consider all directories from HeaderDirs option. Stop at first found.
  for (const StringRef HeaderDir : HeaderDirs) {
    const size_t PosHeaderDir = AbsPath.rfind("/" + HeaderDir.str() + "/");
    if (PosHeaderDir != StringRef::npos) {
      // We don't want the header dir in our guards, i.e. _INCLUDE_
      AbsPath = AbsPath.substr(PosHeaderDir + HeaderDir.size() + 2);
      break; // stop at first found
    }
  }

  std::string Guard = AbsPath;
  llvm::replace(Guard, '/', '_');
  llvm::replace(Guard, '.', '_');
  llvm::replace(Guard, '-', '_');
  Guard = Prefix.str() + Guard;

  return StringRef(Guard).upper();
}

bool HeaderGuardCheck::shouldSuggestEndifComment(StringRef /*Filename*/) {
  return EndifComment;
}

bool HeaderGuardCheck::shouldSuggestToAddHeaderGuard(StringRef Filename) {
  if (AllowPragmaOnce && PragmaOnceLocs.contains(Filename))
    return false;
  return utils::HeaderGuardCheck::shouldSuggestToAddHeaderGuard(Filename);
}

SourceLocation HeaderGuardCheck::getPragmaOnceLoc(StringRef Filename) const {
  const auto It = PragmaOnceLocs.find(Filename);
  return It == PragmaOnceLocs.end() ? SourceLocation() : It->second;
}

void HeaderGuardCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowPragmaOnce", AllowPragmaOnce);
  Options.store(Opts, "EndifComment", EndifComment);
  Options.store(Opts, "HeaderDirs",
                utils::options::serializeStringList(HeaderDirs));
  Options.store(Opts, "Prefix", Prefix);
}

namespace {

class HeaderGuardCallbacks : public PPCallbacks {
public:
  HeaderGuardCallbacks(HeaderGuardCheck *Check, const SourceManager &SM)
      : Check(Check), SM(SM) {}
  void PragmaDirective(SourceLocation Loc,
                       PragmaIntroducerKind /*Introducer*/) override {
    if (!utils::lexer::isPragmaOnce(Loc, SM))
      return;
    if (OptionalFileEntryRef FE = SM.getFileEntryRefForID(SM.getFileID(Loc)))
      Check->PragmaOnceLocs[cleanPath(FE->getName())] = Loc;
    if (!Check->AllowPragmaOnce)
      Check->diag(Loc, "use include guards instead of 'pragma once'");
  }

private:
  HeaderGuardCheck *Check;
  const SourceManager &SM;
};

} // namespace

void HeaderGuardCheck::registerPPCallbacks(const SourceManager &SM,
                                           Preprocessor *PP,
                                           Preprocessor *ModuleExpanderPP) {
  utils::HeaderGuardCheck::registerPPCallbacks(SM, PP, ModuleExpanderPP);
  PP->addPPCallbacks(std::make_unique<HeaderGuardCallbacks>(this, SM));
}

} // namespace clang::tidy::misc
