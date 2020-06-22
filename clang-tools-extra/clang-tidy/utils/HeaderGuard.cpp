//===--- HeaderGuard.cpp - clang-tidy -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeaderGuard.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace tidy {
namespace utils {

/// canonicalize a path by removing ./ and ../ components.
static std::string cleanPath(StringRef Path) {
  SmallString<256> Result = Path;
  llvm::sys::path::remove_dots(Result, true);
  return std::string(Result.str());
}

namespace {
class HeaderGuardPPCallbacks : public PPCallbacks {
public:
  HeaderGuardPPCallbacks(Preprocessor *PP, HeaderGuardCheck *Check)
      : PP(PP), Check(Check) {}

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override {
    // Record all files we enter. We'll need them to diagnose headers without
    // guards.
    SourceManager &SM = PP->getSourceManager();
    if (Reason == EnterFile && FileType == SrcMgr::C_User) {
      if (const FileEntry *FE = SM.getFileEntryForID(SM.getFileID(Loc))) {
        std::string FileName = cleanPath(FE->getName());
        Files[FileName] = FE;
      }
    }
  }

  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override {
    if (MD)
      return;

    // Record #ifndefs that succeeded. We also need the Location of the Name.
    Ifndefs[MacroNameTok.getIdentifierInfo()] =
        std::make_pair(Loc, MacroNameTok.getLocation());
  }

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    // Record all defined macros. We store the whole token to get info on the
    // name later.
    Macros.emplace_back(MacroNameTok, MD->getMacroInfo());
  }

  void Endif(SourceLocation Loc, SourceLocation IfLoc) override {
    // Record all #endif and the corresponding #ifs (including #ifndefs).
    EndIfs[IfLoc] = Loc;
  }

  void EndOfMainFile() override {
    // Now that we have all this information from the preprocessor, use it!
    SourceManager &SM = PP->getSourceManager();

    for (const auto &MacroEntry : Macros) {
      const MacroInfo *MI = MacroEntry.second;

      // We use clang's header guard detection. This has the advantage of also
      // emitting a warning for cases where a pseudo header guard is found but
      // preceded by something blocking the header guard optimization.
      if (!MI->isUsedForHeaderGuard())
        continue;

      const FileEntry *FE =
          SM.getFileEntryForID(SM.getFileID(MI->getDefinitionLoc()));
      std::string FileName = cleanPath(FE->getName());
      Files.erase(FileName);

      // See if we should check and fix this header guard.
      if (!Check->shouldFixHeaderGuard(FileName))
        continue;

      // Look up Locations for this guard.
      SourceLocation Ifndef =
          Ifndefs[MacroEntry.first.getIdentifierInfo()].second;
      SourceLocation Define = MacroEntry.first.getLocation();
      SourceLocation EndIf =
          EndIfs[Ifndefs[MacroEntry.first.getIdentifierInfo()].first];

      // If the macro Name is not equal to what we can compute, correct it in
      // the #ifndef and #define.
      StringRef CurHeaderGuard =
          MacroEntry.first.getIdentifierInfo()->getName();
      std::vector<FixItHint> FixIts;
      std::string NewGuard = checkHeaderGuardDefinition(
          Ifndef, Define, EndIf, FileName, CurHeaderGuard, FixIts);

      // Now look at the #endif. We want a comment with the header guard. Fix it
      // at the slightest deviation.
      checkEndifComment(FileName, EndIf, NewGuard, FixIts);

      // Bundle all fix-its into one warning. The message depends on whether we
      // changed the header guard or not.
      if (!FixIts.empty()) {
        if (CurHeaderGuard != NewGuard) {
          Check->diag(Ifndef, "header guard does not follow preferred style")
              << FixIts;
        } else {
          Check->diag(EndIf, "#endif for a header guard should reference the "
                             "guard macro in a comment")
              << FixIts;
        }
      }
    }

    // Emit warnings for headers that are missing guards.
    checkGuardlessHeaders();

    // Clear all state.
    Macros.clear();
    Files.clear();
    Ifndefs.clear();
    EndIfs.clear();
  }

  bool wouldFixEndifComment(StringRef FileName, SourceLocation EndIf,
                            StringRef HeaderGuard,
                            size_t *EndIfLenPtr = nullptr) {
    if (!EndIf.isValid())
      return false;
    const char *EndIfData = PP->getSourceManager().getCharacterData(EndIf);
    size_t EndIfLen = std::strcspn(EndIfData, "\r\n");
    if (EndIfLenPtr)
      *EndIfLenPtr = EndIfLen;

    StringRef EndIfStr(EndIfData, EndIfLen);
    EndIfStr = EndIfStr.substr(EndIfStr.find_first_not_of("#endif \t"));

    // Give up if there's an escaped newline.
    size_t FindEscapedNewline = EndIfStr.find_last_not_of(' ');
    if (FindEscapedNewline != StringRef::npos &&
        EndIfStr[FindEscapedNewline] == '\\')
      return false;

    if (!Check->shouldSuggestEndifComment(FileName) &&
        !(EndIfStr.startswith("//") ||
          (EndIfStr.startswith("/*") && EndIfStr.endswith("*/"))))
      return false;

    return (EndIfStr != "// " + HeaderGuard.str()) &&
           (EndIfStr != "/* " + HeaderGuard.str() + " */");
  }

  /// Look for header guards that don't match the preferred style. Emit
  /// fix-its and return the suggested header guard (or the original if no
  /// change was made.
  std::string checkHeaderGuardDefinition(SourceLocation Ifndef,
                                         SourceLocation Define,
                                         SourceLocation EndIf,
                                         StringRef FileName,
                                         StringRef CurHeaderGuard,
                                         std::vector<FixItHint> &FixIts) {
    std::string CPPVar = Check->getHeaderGuard(FileName, CurHeaderGuard);
    std::string CPPVarUnder = CPPVar + '_';

    // Allow a trailing underscore iff we don't have to change the endif comment
    // too.
    if (Ifndef.isValid() && CurHeaderGuard != CPPVar &&
        (CurHeaderGuard != CPPVarUnder ||
         wouldFixEndifComment(FileName, EndIf, CurHeaderGuard))) {
      FixIts.push_back(FixItHint::CreateReplacement(
          CharSourceRange::getTokenRange(
              Ifndef, Ifndef.getLocWithOffset(CurHeaderGuard.size())),
          CPPVar));
      FixIts.push_back(FixItHint::CreateReplacement(
          CharSourceRange::getTokenRange(
              Define, Define.getLocWithOffset(CurHeaderGuard.size())),
          CPPVar));
      return CPPVar;
    }
    return std::string(CurHeaderGuard);
  }

  /// Checks the comment after the #endif of a header guard and fixes it
  /// if it doesn't match \c HeaderGuard.
  void checkEndifComment(StringRef FileName, SourceLocation EndIf,
                         StringRef HeaderGuard,
                         std::vector<FixItHint> &FixIts) {
    size_t EndIfLen;
    if (wouldFixEndifComment(FileName, EndIf, HeaderGuard, &EndIfLen)) {
      FixIts.push_back(FixItHint::CreateReplacement(
          CharSourceRange::getCharRange(EndIf,
                                        EndIf.getLocWithOffset(EndIfLen)),
          Check->formatEndIf(HeaderGuard)));
    }
  }

  /// Looks for files that were visited but didn't have a header guard.
  /// Emits a warning with fixits suggesting adding one.
  void checkGuardlessHeaders() {
    // Look for header files that didn't have a header guard. Emit a warning and
    // fix-its to add the guard.
    // TODO: Insert the guard after top comments.
    for (const auto &FE : Files) {
      StringRef FileName = FE.getKey();
      if (!Check->shouldSuggestToAddHeaderGuard(FileName))
        continue;

      SourceManager &SM = PP->getSourceManager();
      FileID FID = SM.translateFile(FE.getValue());
      SourceLocation StartLoc = SM.getLocForStartOfFile(FID);
      if (StartLoc.isInvalid())
        continue;

      std::string CPPVar = Check->getHeaderGuard(FileName);
      std::string CPPVarUnder = CPPVar + '_'; // Allow a trailing underscore.
      // If there's a macro with a name that follows the header guard convention
      // but was not recognized by the preprocessor as a header guard there must
      // be code outside of the guarded area. Emit a plain warning without
      // fix-its.
      // FIXME: Can we move it into the right spot?
      bool SeenMacro = false;
      for (const auto &MacroEntry : Macros) {
        StringRef Name = MacroEntry.first.getIdentifierInfo()->getName();
        SourceLocation DefineLoc = MacroEntry.first.getLocation();
        if ((Name == CPPVar || Name == CPPVarUnder) &&
            SM.isWrittenInSameFile(StartLoc, DefineLoc)) {
          Check->diag(DefineLoc, "code/includes outside of area guarded by "
                                 "header guard; consider moving it");
          SeenMacro = true;
          break;
        }
      }

      if (SeenMacro)
        continue;

      Check->diag(StartLoc, "header is missing header guard")
          << FixItHint::CreateInsertion(
                 StartLoc, "#ifndef " + CPPVar + "\n#define " + CPPVar + "\n\n")
          << FixItHint::CreateInsertion(
                 SM.getLocForEndOfFile(FID),
                 Check->shouldSuggestEndifComment(FileName)
                     ? "\n#" + Check->formatEndIf(CPPVar) + "\n"
                     : "\n#endif\n");
    }
  }

private:
  std::vector<std::pair<Token, const MacroInfo *>> Macros;
  llvm::StringMap<const FileEntry *> Files;
  std::map<const IdentifierInfo *, std::pair<SourceLocation, SourceLocation>>
      Ifndefs;
  std::map<SourceLocation, SourceLocation> EndIfs;

  Preprocessor *PP;
  HeaderGuardCheck *Check;
};
} // namespace

void HeaderGuardCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "HeaderFileExtensions", RawStringHeaderFileExtensions);
}

void HeaderGuardCheck::registerPPCallbacks(const SourceManager &SM,
                                           Preprocessor *PP,
                                           Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(std::make_unique<HeaderGuardPPCallbacks>(PP, this));
}

bool HeaderGuardCheck::shouldSuggestEndifComment(StringRef FileName) {
  return utils::isFileExtension(FileName, HeaderFileExtensions);
}

bool HeaderGuardCheck::shouldFixHeaderGuard(StringRef FileName) { return true; }

bool HeaderGuardCheck::shouldSuggestToAddHeaderGuard(StringRef FileName) {
  return utils::isFileExtension(FileName, HeaderFileExtensions);
}

std::string HeaderGuardCheck::formatEndIf(StringRef HeaderGuard) {
  return "endif // " + HeaderGuard.str();
}
} // namespace utils
} // namespace tidy
} // namespace clang
