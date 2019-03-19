//===--- PPCallbacksTracker.cpp - Preprocessor tracker -*--*---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implementations for preprocessor tracking.
///
/// See the header for details.
///
//===----------------------------------------------------------------------===//

#include "PPCallbacksTracker.h"
#include "clang/Lex/MacroArgs.h"
#include "llvm/Support/raw_ostream.h"

// Utility functions.

// Get a "file:line:column" source location string.
static std::string getSourceLocationString(clang::Preprocessor &PP,
                                           clang::SourceLocation Loc) {
  if (Loc.isInvalid())
    return std::string("(none)");

  if (Loc.isFileID()) {
    clang::PresumedLoc PLoc = PP.getSourceManager().getPresumedLoc(Loc);

    if (PLoc.isInvalid()) {
      return std::string("(invalid)");
    }

    std::string Str;
    llvm::raw_string_ostream SS(Str);

    // The macro expansion and spelling pos is identical for file locs.
    SS << "\"" << PLoc.getFilename() << ':' << PLoc.getLine() << ':'
       << PLoc.getColumn() << "\"";

    std::string Result = SS.str();

    // YAML treats backslash as escape, so use forward slashes.
    std::replace(Result.begin(), Result.end(), '\\', '/');

    return Result;
  }

  return std::string("(nonfile)");
}

// Enum string tables.

// FileChangeReason strings.
static const char *const FileChangeReasonStrings[] = {
  "EnterFile", "ExitFile", "SystemHeaderPragma", "RenameFile"
};

// CharacteristicKind strings.
static const char *const CharacteristicKindStrings[] = { "C_User", "C_System",
                                                         "C_ExternCSystem" };

// MacroDirective::Kind strings.
static const char *const MacroDirectiveKindStrings[] = {
  "MD_Define","MD_Undefine", "MD_Visibility"
};

// PragmaIntroducerKind strings.
static const char *const PragmaIntroducerKindStrings[] = { "PIK_HashPragma",
                                                           "PIK__Pragma",
                                                           "PIK___pragma" };

// PragmaMessageKind strings.
static const char *const PragmaMessageKindStrings[] = {
  "PMK_Message", "PMK_Warning", "PMK_Error"
};

// ConditionValueKind strings.
static const char *const ConditionValueKindStrings[] = {
  "CVK_NotEvaluated", "CVK_False", "CVK_True"
};

// Mapping strings.
static const char *const MappingStrings[] = { "0",          "MAP_IGNORE",
                                              "MAP_REMARK", "MAP_WARNING",
                                              "MAP_ERROR",  "MAP_FATAL" };

// PPCallbacksTracker functions.

PPCallbacksTracker::PPCallbacksTracker(const FilterType &Filters,
                                       std::vector<CallbackCall> &CallbackCalls,
                                       clang::Preprocessor &PP)
    : CallbackCalls(CallbackCalls), Filters(Filters), PP(PP) {}

PPCallbacksTracker::~PPCallbacksTracker() {}

// Callback functions.

// Callback invoked whenever a source file is entered or exited.
void PPCallbacksTracker::FileChanged(
    clang::SourceLocation Loc, clang::PPCallbacks::FileChangeReason Reason,
    clang::SrcMgr::CharacteristicKind FileType, clang::FileID PrevFID) {
  beginCallback("FileChanged");
  appendArgument("Loc", Loc);
  appendArgument("Reason", Reason, FileChangeReasonStrings);
  appendArgument("FileType", FileType, CharacteristicKindStrings);
  appendArgument("PrevFID", PrevFID);
}

// Callback invoked whenever a source file is skipped as the result
// of header guard optimization.
void
PPCallbacksTracker::FileSkipped(const clang::FileEntry &SkippedFile,
                                const clang::Token &FilenameTok,
                                clang::SrcMgr::CharacteristicKind FileType) {
  beginCallback("FileSkipped");
  appendArgument("ParentFile", &SkippedFile);
  appendArgument("FilenameTok", FilenameTok);
  appendArgument("FileType", FileType, CharacteristicKindStrings);
}

// Callback invoked whenever an inclusion directive results in a
// file-not-found error.
bool
PPCallbacksTracker::FileNotFound(llvm::StringRef FileName,
                                 llvm::SmallVectorImpl<char> &RecoveryPath) {
  beginCallback("FileNotFound");
  appendFilePathArgument("FileName", FileName);
  return false;
}

// Callback invoked whenever an inclusion directive of
// any kind (#include, #import, etc.) has been processed, regardless
// of whether the inclusion will actually result in an inclusion.
void PPCallbacksTracker::InclusionDirective(
    clang::SourceLocation HashLoc, const clang::Token &IncludeTok,
    llvm::StringRef FileName, bool IsAngled,
    clang::CharSourceRange FilenameRange, const clang::FileEntry *File,
    llvm::StringRef SearchPath, llvm::StringRef RelativePath,
    const clang::Module *Imported, clang::SrcMgr::CharacteristicKind FileType) {
  beginCallback("InclusionDirective");
  appendArgument("IncludeTok", IncludeTok);
  appendFilePathArgument("FileName", FileName);
  appendArgument("IsAngled", IsAngled);
  appendArgument("FilenameRange", FilenameRange);
  appendArgument("File", File);
  appendFilePathArgument("SearchPath", SearchPath);
  appendFilePathArgument("RelativePath", RelativePath);
  appendArgument("Imported", Imported);
}

// Callback invoked whenever there was an explicit module-import
// syntax.
void PPCallbacksTracker::moduleImport(clang::SourceLocation ImportLoc,
                                      clang::ModuleIdPath Path,
                                      const clang::Module *Imported) {
  beginCallback("moduleImport");
  appendArgument("ImportLoc", ImportLoc);
  appendArgument("Path", Path);
  appendArgument("Imported", Imported);
}

// Callback invoked when the end of the main file is reached.
// No subsequent callbacks will be made.
void PPCallbacksTracker::EndOfMainFile() { beginCallback("EndOfMainFile"); }

// Callback invoked when a #ident or #sccs directive is read.
void PPCallbacksTracker::Ident(clang::SourceLocation Loc, llvm::StringRef Str) {
  beginCallback("Ident");
  appendArgument("Loc", Loc);
  appendArgument("Str", Str);
}

// Callback invoked when start reading any pragma directive.
void
PPCallbacksTracker::PragmaDirective(clang::SourceLocation Loc,
                                    clang::PragmaIntroducerKind Introducer) {
  beginCallback("PragmaDirective");
  appendArgument("Loc", Loc);
  appendArgument("Introducer", Introducer, PragmaIntroducerKindStrings);
}

// Callback invoked when a #pragma comment directive is read.
void PPCallbacksTracker::PragmaComment(clang::SourceLocation Loc,
                                       const clang::IdentifierInfo *Kind,
                                       llvm::StringRef Str) {
  beginCallback("PragmaComment");
  appendArgument("Loc", Loc);
  appendArgument("Kind", Kind);
  appendArgument("Str", Str);
}

// Callback invoked when a #pragma detect_mismatch directive is
// read.
void PPCallbacksTracker::PragmaDetectMismatch(clang::SourceLocation Loc,
                                              llvm::StringRef Name,
                                              llvm::StringRef Value) {
  beginCallback("PragmaDetectMismatch");
  appendArgument("Loc", Loc);
  appendArgument("Name", Name);
  appendArgument("Value", Value);
}

// Callback invoked when a #pragma clang __debug directive is read.
void PPCallbacksTracker::PragmaDebug(clang::SourceLocation Loc,
                                     llvm::StringRef DebugType) {
  beginCallback("PragmaDebug");
  appendArgument("Loc", Loc);
  appendArgument("DebugType", DebugType);
}

// Callback invoked when a #pragma message directive is read.
void PPCallbacksTracker::PragmaMessage(
    clang::SourceLocation Loc, llvm::StringRef Namespace,
    clang::PPCallbacks::PragmaMessageKind Kind, llvm::StringRef Str) {
  beginCallback("PragmaMessage");
  appendArgument("Loc", Loc);
  appendArgument("Namespace", Namespace);
  appendArgument("Kind", Kind, PragmaMessageKindStrings);
  appendArgument("Str", Str);
}

// Callback invoked when a #pragma gcc dianostic push directive
// is read.
void PPCallbacksTracker::PragmaDiagnosticPush(clang::SourceLocation Loc,
                                              llvm::StringRef Namespace) {
  beginCallback("PragmaDiagnosticPush");
  appendArgument("Loc", Loc);
  appendArgument("Namespace", Namespace);
}

// Callback invoked when a #pragma gcc dianostic pop directive
// is read.
void PPCallbacksTracker::PragmaDiagnosticPop(clang::SourceLocation Loc,
                                             llvm::StringRef Namespace) {
  beginCallback("PragmaDiagnosticPop");
  appendArgument("Loc", Loc);
  appendArgument("Namespace", Namespace);
}

// Callback invoked when a #pragma gcc dianostic directive is read.
void PPCallbacksTracker::PragmaDiagnostic(clang::SourceLocation Loc,
                                          llvm::StringRef Namespace,
                                          clang::diag::Severity Mapping,
                                          llvm::StringRef Str) {
  beginCallback("PragmaDiagnostic");
  appendArgument("Loc", Loc);
  appendArgument("Namespace", Namespace);
  appendArgument("Mapping", (unsigned)Mapping, MappingStrings);
  appendArgument("Str", Str);
}

// Called when an OpenCL extension is either disabled or
// enabled with a pragma.
void PPCallbacksTracker::PragmaOpenCLExtension(
    clang::SourceLocation NameLoc, const clang::IdentifierInfo *Name,
    clang::SourceLocation StateLoc, unsigned State) {
  beginCallback("PragmaOpenCLExtension");
  appendArgument("NameLoc", NameLoc);
  appendArgument("Name", Name);
  appendArgument("StateLoc", StateLoc);
  appendArgument("State", (int)State);
}

// Callback invoked when a #pragma warning directive is read.
void PPCallbacksTracker::PragmaWarning(clang::SourceLocation Loc,
                                       llvm::StringRef WarningSpec,
                                       llvm::ArrayRef<int> Ids) {
  beginCallback("PragmaWarning");
  appendArgument("Loc", Loc);
  appendArgument("WarningSpec", WarningSpec);

  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << "[";
  for (int i = 0, e = Ids.size(); i != e; ++i) {
    if (i)
      SS << ", ";
    SS << Ids[i];
  }
  SS << "]";
  appendArgument("Ids", SS.str());
}

// Callback invoked when a #pragma warning(push) directive is read.
void PPCallbacksTracker::PragmaWarningPush(clang::SourceLocation Loc,
                                           int Level) {
  beginCallback("PragmaWarningPush");
  appendArgument("Loc", Loc);
  appendArgument("Level", Level);
}

// Callback invoked when a #pragma warning(pop) directive is read.
void PPCallbacksTracker::PragmaWarningPop(clang::SourceLocation Loc) {
  beginCallback("PragmaWarningPop");
  appendArgument("Loc", Loc);
}

// Callback invoked when a #pragma execution_character_set(push) directive
// is read.
void PPCallbacksTracker::PragmaExecCharsetPush(clang::SourceLocation Loc,
                                               clang::StringRef Str) {
  beginCallback("PragmaExecCharsetPush");
  appendArgument("Loc", Loc);
  appendArgument("Charset", Str);
}

// Callback invoked when a #pragma execution_character_set(pop) directive
// is read.
void PPCallbacksTracker::PragmaExecCharsetPop(clang::SourceLocation Loc) {
  beginCallback("PragmaExecCharsetPop");
  appendArgument("Loc", Loc);
}

// Called by Preprocessor::HandleMacroExpandedIdentifier when a
// macro invocation is found.
void
PPCallbacksTracker::MacroExpands(const clang::Token &MacroNameTok,
                                 const clang::MacroDefinition &MacroDefinition,
                                 clang::SourceRange Range,
                                 const clang::MacroArgs *Args) {
  beginCallback("MacroExpands");
  appendArgument("MacroNameTok", MacroNameTok);
  appendArgument("MacroDefinition", MacroDefinition);
  appendArgument("Range", Range);
  appendArgument("Args", Args);
}

// Hook called whenever a macro definition is seen.
void
PPCallbacksTracker::MacroDefined(const clang::Token &MacroNameTok,
                                 const clang::MacroDirective *MacroDirective) {
  beginCallback("MacroDefined");
  appendArgument("MacroNameTok", MacroNameTok);
  appendArgument("MacroDirective", MacroDirective);
}

// Hook called whenever a macro #undef is seen.
void PPCallbacksTracker::MacroUndefined(
    const clang::Token &MacroNameTok,
    const clang::MacroDefinition &MacroDefinition,
    const clang::MacroDirective *Undef) {
  beginCallback("MacroUndefined");
  appendArgument("MacroNameTok", MacroNameTok);
  appendArgument("MacroDefinition", MacroDefinition);
}

// Hook called whenever the 'defined' operator is seen.
void PPCallbacksTracker::Defined(const clang::Token &MacroNameTok,
                                 const clang::MacroDefinition &MacroDefinition,
                                 clang::SourceRange Range) {
  beginCallback("Defined");
  appendArgument("MacroNameTok", MacroNameTok);
  appendArgument("MacroDefinition", MacroDefinition);
  appendArgument("Range", Range);
}

// Hook called when a source range is skipped.
void PPCallbacksTracker::SourceRangeSkipped(clang::SourceRange Range,
                                            clang::SourceLocation EndifLoc) {
  beginCallback("SourceRangeSkipped");
  appendArgument("Range", clang::SourceRange(Range.getBegin(), EndifLoc));
}

// Hook called whenever an #if is seen.
void PPCallbacksTracker::If(clang::SourceLocation Loc,
                            clang::SourceRange ConditionRange,
                            ConditionValueKind ConditionValue) {
  beginCallback("If");
  appendArgument("Loc", Loc);
  appendArgument("ConditionRange", ConditionRange);
  appendArgument("ConditionValue", ConditionValue, ConditionValueKindStrings);
}

// Hook called whenever an #elif is seen.
void PPCallbacksTracker::Elif(clang::SourceLocation Loc,
                              clang::SourceRange ConditionRange,
                              ConditionValueKind ConditionValue,
                              clang::SourceLocation IfLoc) {
  beginCallback("Elif");
  appendArgument("Loc", Loc);
  appendArgument("ConditionRange", ConditionRange);
  appendArgument("ConditionValue", ConditionValue, ConditionValueKindStrings);
  appendArgument("IfLoc", IfLoc);
}

// Hook called whenever an #ifdef is seen.
void PPCallbacksTracker::Ifdef(clang::SourceLocation Loc,
                               const clang::Token &MacroNameTok,
                               const clang::MacroDefinition &MacroDefinition) {
  beginCallback("Ifdef");
  appendArgument("Loc", Loc);
  appendArgument("MacroNameTok", MacroNameTok);
  appendArgument("MacroDefinition", MacroDefinition);
}

// Hook called whenever an #ifndef is seen.
void PPCallbacksTracker::Ifndef(clang::SourceLocation Loc,
                                const clang::Token &MacroNameTok,
                                const clang::MacroDefinition &MacroDefinition) {
  beginCallback("Ifndef");
  appendArgument("Loc", Loc);
  appendArgument("MacroNameTok", MacroNameTok);
  appendArgument("MacroDefinition", MacroDefinition);
}

// Hook called whenever an #else is seen.
void PPCallbacksTracker::Else(clang::SourceLocation Loc,
                              clang::SourceLocation IfLoc) {
  beginCallback("Else");
  appendArgument("Loc", Loc);
  appendArgument("IfLoc", IfLoc);
}

// Hook called whenever an #endif is seen.
void PPCallbacksTracker::Endif(clang::SourceLocation Loc,
                               clang::SourceLocation IfLoc) {
  beginCallback("Endif");
  appendArgument("Loc", Loc);
  appendArgument("IfLoc", IfLoc);
}

// Helper functions.

// Start a new callback.
void PPCallbacksTracker::beginCallback(const char *Name) {
  auto R = CallbackIsEnabled.try_emplace(Name, false);
  if (R.second) {
    llvm::StringRef N(Name);
    for (const std::pair<llvm::GlobPattern, bool> &Filter : Filters)
      if (Filter.first.match(N))
        R.first->second = Filter.second;
  }
  DisableTrace = !R.first->second;
  if (DisableTrace)
    return;
  CallbackCalls.push_back(CallbackCall(Name));
}

// Append a bool argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name, bool Value) {
  appendArgument(Name, (Value ? "true" : "false"));
}

// Append an int argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name, int Value) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << Value;
  appendArgument(Name, SS.str());
}

// Append a string argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name, const char *Value) {
  if (DisableTrace)
    return;
  CallbackCalls.back().Arguments.push_back(Argument(Name, Value));
}

// Append a string object argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        llvm::StringRef Value) {
  appendArgument(Name, Value.str());
}

// Append a string object argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        const std::string &Value) {
  appendArgument(Name, Value.c_str());
}

// Append a token argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        const clang::Token &Value) {
  appendArgument(Name, PP.getSpelling(Value));
}

// Append an enum argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name, int Value,
                                        const char *const Strings[]) {
  appendArgument(Name, Strings[Value]);
}

// Append a FileID argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name, clang::FileID Value) {
  if (Value.isInvalid()) {
    appendArgument(Name, "(invalid)");
    return;
  }
  const clang::FileEntry *FileEntry =
      PP.getSourceManager().getFileEntryForID(Value);
  if (!FileEntry) {
    appendArgument(Name, "(getFileEntryForID failed)");
    return;
  }
  appendFilePathArgument(Name, FileEntry->getName());
}

// Append a FileEntry argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        const clang::FileEntry *Value) {
  if (!Value) {
    appendArgument(Name, "(null)");
    return;
  }
  appendFilePathArgument(Name, Value->getName());
}

// Append a SourceLocation argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        clang::SourceLocation Value) {
  if (Value.isInvalid()) {
    appendArgument(Name, "(invalid)");
    return;
  }
  appendArgument(Name, getSourceLocationString(PP, Value).c_str());
}

// Append a SourceRange argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        clang::SourceRange Value) {
  if (DisableTrace)
    return;
  if (Value.isInvalid()) {
    appendArgument(Name, "(invalid)");
    return;
  }
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << "[" << getSourceLocationString(PP, Value.getBegin()) << ", "
     << getSourceLocationString(PP, Value.getEnd()) << "]";
  appendArgument(Name, SS.str());
}

// Append a CharSourceRange argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        clang::CharSourceRange Value) {
  if (Value.isInvalid()) {
    appendArgument(Name, "(invalid)");
    return;
  }
  appendArgument(Name, getSourceString(Value).str().c_str());
}

// Append a SourceLocation argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        clang::ModuleIdPath Value) {
  if (DisableTrace)
    return;
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << "[";
  for (int I = 0, E = Value.size(); I != E; ++I) {
    if (I)
      SS << ", ";
    SS << "{"
       << "Name: " << Value[I].first->getName() << ", "
       << "Loc: " << getSourceLocationString(PP, Value[I].second) << "}";
  }
  SS << "]";
  appendArgument(Name, SS.str());
}

// Append an IdentifierInfo argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        const clang::IdentifierInfo *Value) {
  if (!Value) {
    appendArgument(Name, "(null)");
    return;
  }
  appendArgument(Name, Value->getName().str().c_str());
}

// Append a MacroDirective argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        const clang::MacroDirective *Value) {
  if (!Value) {
    appendArgument(Name, "(null)");
    return;
  }
  appendArgument(Name, MacroDirectiveKindStrings[Value->getKind()]);
}

// Append a MacroDefinition argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        const clang::MacroDefinition &Value) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << "[";
  bool Any = false;
  if (Value.getLocalDirective()) {
    SS << "(local)";
    Any = true;
  }
  for (auto *MM : Value.getModuleMacros()) {
    if (Any) SS << ", ";
    SS << MM->getOwningModule()->getFullModuleName();
  }
  SS << "]";
  appendArgument(Name, SS.str());
}

// Append a MacroArgs argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        const clang::MacroArgs *Value) {
  if (!Value) {
    appendArgument(Name, "(null)");
    return;
  }
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << "[";

  // Each argument is is a series of contiguous Tokens, terminated by a eof.
  // Go through each argument printing tokens until we reach eof.
  for (unsigned I = 0; I < Value->getNumMacroArguments(); ++I) {
    const clang::Token *Current = Value->getUnexpArgument(I);
    if (I)
      SS << ", ";
    bool First = true;
    while (Current->isNot(clang::tok::eof)) {
      if (!First)
        SS << " ";
      // We need to be careful here because the arguments might not be legal in
      // YAML, so we use the token name for anything but identifiers and
      // numeric literals.
      if (Current->isAnyIdentifier() ||
          Current->is(clang::tok::numeric_constant)) {
        SS << PP.getSpelling(*Current);
      } else {
        SS << "<" << Current->getName() << ">";
      }
      ++Current;
      First = false;
    }
  }
  SS << "]";
  appendArgument(Name, SS.str());
}

// Append a Module argument to the top trace item.
void PPCallbacksTracker::appendArgument(const char *Name,
                                        const clang::Module *Value) {
  if (!Value) {
    appendArgument(Name, "(null)");
    return;
  }
  appendArgument(Name, Value->Name.c_str());
}

// Append a double-quoted argument to the top trace item.
void PPCallbacksTracker::appendQuotedArgument(const char *Name,
                                              const std::string &Value) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << "\"" << Value << "\"";
  appendArgument(Name, SS.str());
}

// Append a double-quoted file path argument to the top trace item.
void PPCallbacksTracker::appendFilePathArgument(const char *Name,
                                                llvm::StringRef Value) {
  std::string Path(Value);
  // YAML treats backslash as escape, so use forward slashes.
  std::replace(Path.begin(), Path.end(), '\\', '/');
  appendQuotedArgument(Name, Path);
}

// Get the raw source string of the range.
llvm::StringRef
PPCallbacksTracker::getSourceString(clang::CharSourceRange Range) {
  const char *B = PP.getSourceManager().getCharacterData(Range.getBegin());
  const char *E = PP.getSourceManager().getCharacterData(Range.getEnd());
  return llvm::StringRef(B, E - B);
}
