//===--- PPCallbacksTracker.h - Preprocessor tracking -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Classes and definitions for preprocessor tracking.
///
/// The core definition is the PPCallbacksTracker class, derived from Clang's
/// PPCallbacks class from the Lex library, which overrides all the callbacks
/// and collects information about each callback call, saving it in a
/// data structure built up of CallbackCall and Argument objects, which
/// record the preprocessor callback name and arguments in high-level string
/// form for later inspection.
///
//===----------------------------------------------------------------------===//

#ifndef PPTRACE_PPCALLBACKSTRACKER_H
#define PPTRACE_PPCALLBACKSTRACKER_H

#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/GlobPattern.h"
#include <string>
#include <vector>

namespace clang {
namespace pp_trace {

// This struct represents one callback function argument by name and value.
struct Argument {
  std::string Name;
  std::string Value;
};

/// \brief This class represents one callback call by name and an array
///   of arguments.
class CallbackCall {
public:
  CallbackCall(llvm::StringRef Name) : Name(Name) {}
  CallbackCall() = default;

  std::string Name;
  std::vector<Argument> Arguments;
};

using FilterType = std::vector<std::pair<llvm::GlobPattern, bool>>;

/// \brief This class overrides the PPCallbacks class for tracking preprocessor
///   activity by means of its callback functions.
///
/// This object is given a vector for storing the trace information, built up
/// of CallbackCall and subordinate Argument objects for representing the
/// callback calls and their arguments.  It's a reference so the vector can
/// exist beyond the lifetime of this object, because it's deleted by the
/// preprocessor automatically in its destructor.
///
/// This class supports a mechanism for inhibiting trace output for
/// specific callbacks by name, for the purpose of eliminating output for
/// callbacks of no interest that might clutter the output.
///
/// Following the constructor and destructor function declarations, the
/// overidden callback functions are defined.  The remaining functions are
/// helpers for recording the trace data, to reduce the coupling between it
/// and the recorded data structure.
class PPCallbacksTracker : public PPCallbacks {
public:
  /// \brief Note that all of the arguments are references, and owned
  /// by the caller.
  /// \param Filters - List of (Glob,Enabled) pairs used to filter callbacks.
  /// \param CallbackCalls - Trace buffer.
  /// \param PP - The preprocessor.  Needed for getting some argument strings.
  PPCallbacksTracker(const FilterType &Filters,
                     std::vector<CallbackCall> &CallbackCalls,
                     Preprocessor &PP);

  ~PPCallbacksTracker() override;

  // Overidden callback functions.

  void FileChanged(SourceLocation Loc, PPCallbacks::FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID = FileID()) override;
  void FileSkipped(const FileEntry &SkippedFile, const Token &FilenameTok,
                   SrcMgr::CharacteristicKind FileType) override;
  bool FileNotFound(llvm::StringRef FileName,
                    llvm::SmallVectorImpl<char> &RecoveryPath) override;
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          llvm::StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          llvm::StringRef SearchPath,
                          llvm::StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;
  void moduleImport(SourceLocation ImportLoc, ModuleIdPath Path,
                    const Module *Imported) override;
  void EndOfMainFile() override;
  void Ident(SourceLocation Loc, llvm::StringRef str) override;
  void PragmaDirective(SourceLocation Loc,
                       PragmaIntroducerKind Introducer) override;
  void PragmaComment(SourceLocation Loc, const IdentifierInfo *Kind,
                     llvm::StringRef Str) override;
  void PragmaDetectMismatch(SourceLocation Loc, llvm::StringRef Name,
                            llvm::StringRef Value) override;
  void PragmaDebug(SourceLocation Loc, llvm::StringRef DebugType) override;
  void PragmaMessage(SourceLocation Loc, llvm::StringRef Namespace,
                     PPCallbacks::PragmaMessageKind Kind,
                     llvm::StringRef Str) override;
  void PragmaDiagnosticPush(SourceLocation Loc,
                            llvm::StringRef Namespace) override;
  void PragmaDiagnosticPop(SourceLocation Loc,
                           llvm::StringRef Namespace) override;
  void PragmaDiagnostic(SourceLocation Loc, llvm::StringRef Namespace,
                        diag::Severity mapping, llvm::StringRef Str) override;
  void PragmaOpenCLExtension(SourceLocation NameLoc, const IdentifierInfo *Name,
                             SourceLocation StateLoc, unsigned State) override;
  void PragmaWarning(SourceLocation Loc, llvm::StringRef WarningSpec,
                     llvm::ArrayRef<int> Ids) override;
  void PragmaWarningPush(SourceLocation Loc, int Level) override;
  void PragmaWarningPop(SourceLocation Loc) override;
  void PragmaExecCharsetPush(SourceLocation Loc, StringRef Str) override;
  void PragmaExecCharsetPop(SourceLocation Loc) override;
  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override;
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override;
  void MacroUndefined(const Token &MacroNameTok, const MacroDefinition &MD,
                      const MacroDirective *Undef) override;
  void Defined(const Token &MacroNameTok, const MacroDefinition &MD,
               SourceRange Range) override;
  void SourceRangeSkipped(SourceRange Range, SourceLocation EndifLoc) override;
  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind ConditionValue) override;
  void Elif(SourceLocation Loc, SourceRange ConditionRange,
            ConditionValueKind ConditionValue, SourceLocation IfLoc) override;
  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MD) override;
  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override;
  void Else(SourceLocation Loc, SourceLocation IfLoc) override;
  void Endif(SourceLocation Loc, SourceLocation IfLoc) override;

  // Helper functions.

  /// \brief Start a new callback.
  void beginCallback(const char *Name);

  /// \brief Append a string to the top trace item.
  void append(const char *Str);

  /// \brief Append a bool argument to the top trace item.
  void appendArgument(const char *Name, bool Value);

  /// \brief Append an int argument to the top trace item.
  void appendArgument(const char *Name, int Value);

  /// \brief Append a string argument to the top trace item.
  void appendArgument(const char *Name, const char *Value);

  /// \brief Append a string reference object argument to the top trace item.
  void appendArgument(const char *Name, llvm::StringRef Value);

  /// \brief Append a string object argument to the top trace item.
  void appendArgument(const char *Name, const std::string &Value);

  /// \brief Append a token argument to the top trace item.
  void appendArgument(const char *Name, const Token &Value);

  /// \brief Append an enum argument to the top trace item.
  void appendArgument(const char *Name, int Value, const char *const Strings[]);

  /// \brief Append a FileID argument to the top trace item.
  void appendArgument(const char *Name, FileID Value);

  /// \brief Append a FileEntry argument to the top trace item.
  void appendArgument(const char *Name, const FileEntry *Value);

  /// \brief Append a SourceLocation argument to the top trace item.
  void appendArgument(const char *Name, SourceLocation Value);

  /// \brief Append a SourceRange argument to the top trace item.
  void appendArgument(const char *Name, SourceRange Value);

  /// \brief Append a CharSourceRange argument to the top trace item.
  void appendArgument(const char *Name, CharSourceRange Value);

  /// \brief Append a ModuleIdPath argument to the top trace item.
  void appendArgument(const char *Name, ModuleIdPath Value);

  /// \brief Append an IdentifierInfo argument to the top trace item.
  void appendArgument(const char *Name, const IdentifierInfo *Value);

  /// \brief Append a MacroDirective argument to the top trace item.
  void appendArgument(const char *Name, const MacroDirective *Value);

  /// \brief Append a MacroDefinition argument to the top trace item.
  void appendArgument(const char *Name, const MacroDefinition &Value);

  /// \brief Append a MacroArgs argument to the top trace item.
  void appendArgument(const char *Name, const MacroArgs *Value);

  /// \brief Append a Module argument to the top trace item.
  void appendArgument(const char *Name, const Module *Value);

  /// \brief Append a double-quoted argument to the top trace item.
  void appendQuotedArgument(const char *Name, const std::string &Value);

  /// \brief Append a double-quoted file path argument to the top trace item.
  void appendFilePathArgument(const char *Name, llvm::StringRef Value);

  /// \brief Get the raw source string of the range.
  llvm::StringRef getSourceString(CharSourceRange Range);

  /// \brief Callback trace information.
  /// We use a reference so the trace will be preserved for the caller
  /// after this object is destructed.
  std::vector<CallbackCall> &CallbackCalls;

  // List of (Glob,Enabled) pairs used to filter callbacks.
  const FilterType &Filters;

  // Whether a callback should be printed.
  llvm::StringMap<bool> CallbackIsEnabled;

  /// \brief Inhibit trace while this is set.
  bool DisableTrace;

  Preprocessor &PP;
};

} // namespace pp_trace
} // namespace clang

#endif // PPTRACE_PPCALLBACKSTRACKER_H
