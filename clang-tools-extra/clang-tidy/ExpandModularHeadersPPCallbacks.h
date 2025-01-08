//===- ExpandModularHeadersPPCallbacks.h - clang-tidy -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_EXPANDMODULARHEADERSPPCALLBACKS_H_
#define LLVM_CLANG_TOOLING_EXPANDMODULARHEADERSPPCALLBACKS_H_

#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm::vfs {
class OverlayFileSystem;
class InMemoryFileSystem;
} // namespace llvm::vfs

namespace clang {
class CompilerInstance;

namespace serialization {
class ModuleFile;
} // namespace serialization

namespace tooling {

/// Handles PPCallbacks and re-runs preprocessing of the whole
/// translation unit with modules disabled.
///
/// This way it's possible to get PPCallbacks for the whole translation unit
/// including the contents of the modular headers and all their transitive
/// includes.
///
/// This allows existing tools based on PPCallbacks to retain their functionality
/// when running with C++ modules enabled. This only works in the backwards
/// compatible modules mode, i.e. when code can still be parsed in non-modular
/// way.
class ExpandModularHeadersPPCallbacks : public PPCallbacks {
public:
  ExpandModularHeadersPPCallbacks(
      CompilerInstance *CI,
      IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS);
  ~ExpandModularHeadersPPCallbacks();

  /// Returns the preprocessor that provides callbacks for the whole
  /// translation unit, including the main file, textual headers, and modular
  /// headers.
  ///
  /// This preprocessor is separate from the one used by the rest of the
  /// compiler.
  Preprocessor *getPreprocessor() const;

private:
  class FileRecorder;

  void handleModuleFile(serialization::ModuleFile *MF);
  void parseToLocation(SourceLocation Loc);

  // Handle PPCallbacks.
  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override;

  void InclusionDirective(SourceLocation DirectiveLoc,
                          const Token &IncludeToken, StringRef IncludedFilename,
                          bool IsAngled, CharSourceRange FilenameRange,
                          OptionalFileEntryRef IncludedFile,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *SuggestedModule, bool ModuleImported,
                          SrcMgr::CharacteristicKind FileType) override;

  void EndOfMainFile() override;

  // Handle all other callbacks.
  // Just parse to the corresponding location to generate PPCallbacks for the
  // corresponding range
  void Ident(SourceLocation Loc, StringRef) override;
  void PragmaDirective(SourceLocation Loc, PragmaIntroducerKind) override;
  void PragmaComment(SourceLocation Loc, const IdentifierInfo *,
                     StringRef) override;
  void PragmaDetectMismatch(SourceLocation Loc, StringRef, StringRef) override;
  void PragmaDebug(SourceLocation Loc, StringRef) override;
  void PragmaMessage(SourceLocation Loc, StringRef, PragmaMessageKind,
                     StringRef) override;
  void PragmaDiagnosticPush(SourceLocation Loc, StringRef) override;
  void PragmaDiagnosticPop(SourceLocation Loc, StringRef) override;
  void PragmaDiagnostic(SourceLocation Loc, StringRef, diag::Severity,
                        StringRef) override;
  void HasInclude(SourceLocation Loc, StringRef, bool, OptionalFileEntryRef,
                  SrcMgr::CharacteristicKind) override;
  void PragmaOpenCLExtension(SourceLocation NameLoc, const IdentifierInfo *,
                             SourceLocation StateLoc, unsigned) override;
  void PragmaWarning(SourceLocation Loc, PragmaWarningSpecifier,
                     ArrayRef<int>) override;
  void PragmaWarningPush(SourceLocation Loc, int) override;
  void PragmaWarningPop(SourceLocation Loc) override;
  void PragmaAssumeNonNullBegin(SourceLocation Loc) override;
  void PragmaAssumeNonNullEnd(SourceLocation Loc) override;
  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &,
                    SourceRange Range, const MacroArgs *) override;
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override;
  void MacroUndefined(const Token &, const MacroDefinition &,
                      const MacroDirective *Undef) override;
  void Defined(const Token &MacroNameTok, const MacroDefinition &,
               SourceRange Range) override;
  void SourceRangeSkipped(SourceRange Range, SourceLocation EndifLoc) override;
  void If(SourceLocation Loc, SourceRange, ConditionValueKind) override;
  void Elif(SourceLocation Loc, SourceRange, ConditionValueKind,
            SourceLocation) override;
  void Ifdef(SourceLocation Loc, const Token &,
             const MacroDefinition &) override;
  void Ifndef(SourceLocation Loc, const Token &,
              const MacroDefinition &) override;
  void Else(SourceLocation Loc, SourceLocation) override;
  void Endif(SourceLocation Loc, SourceLocation) override;

  std::unique_ptr<FileRecorder> Recorder;
  // Set of all the modules visited. Avoids processing a module more than once.
  llvm::DenseSet<serialization::ModuleFile *> VisitedModules;

  CompilerInstance &Compiler;
  // Additional filesystem for replay. Provides all input files from modules.
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFs;

  SourceManager &Sources;
  DiagnosticsEngine Diags;
  LangOptions LangOpts;
  TrivialModuleLoader ModuleLoader;

  std::unique_ptr<HeaderSearch> HeaderInfo;
  std::unique_ptr<Preprocessor> PP;
  bool EnteredMainFile = false;
  bool StartedLexing = false;
  Token CurrentToken;
};

} // namespace tooling
} // namespace clang

#endif // LLVM_CLANG_TOOLING_EXPANDMODULARHEADERSPPCALLBACKS_H_
