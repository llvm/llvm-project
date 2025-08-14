//===--- TrivialDirectiveTracer.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TrivialDirectiveTracer interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_TRIVIAL_DIRECTIVE_TRACER_H
#define LLVM_CLANG_LEX_TRIVIAL_DIRECTIVE_TRACER_H

#include "clang/Lex/PPCallbacks.h"

namespace clang {
class Preprocessor;

class TrivialDirectiveTracer : public PPCallbacks {
  Preprocessor &PP;
  bool InMainFile = true;
  bool SeenNoTrivialPPDirective = false;

  void setSeenNoTrivialPPDirective(bool Val);

public:
  TrivialDirectiveTracer(Preprocessor &P) : PP(P) {}

  bool hasSeenNoTrivialPPDirective() const;

  /// Callback invoked whenever a source file is entered or exited.
  ///
  /// \param Loc Indicates the new location.
  /// \param PrevFID the file that was exited if \p Reason is ExitFile or the
  /// the file before the new one entered for \p Reason EnterFile.
  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID = FileID()) override;

  /// Callback invoked whenever the \p Lexer moves to a different file for
  /// lexing. Unlike \p FileChanged line number directives and other related
  /// pragmas do not trigger callbacks to \p LexedFileChanged.
  ///
  /// \param FID The \p FileID that the \p Lexer moved to.
  ///
  /// \param Reason Whether the \p Lexer entered a new file or exited one.
  ///
  /// \param FileType The \p CharacteristicKind of the file the \p Lexer moved
  /// to.
  ///
  /// \param PrevFID The \p FileID the \p Lexer was using before the change.
  ///
  /// \param Loc The location where the \p Lexer entered a new file from or the
  /// location that the \p Lexer moved into after exiting a file.
  void LexedFileChanged(FileID FID, LexedFileChangeReason Reason,
                        SrcMgr::CharacteristicKind FileType, FileID PrevFID,
                        SourceLocation Loc) override;

  /// Callback invoked whenever an embed directive has been processed,
  /// regardless of whether the embed will actually find a file.
  ///
  /// \param HashLoc The location of the '#' that starts the embed directive.
  ///
  /// \param FileName The name of the file being included, as written in the
  /// source code.
  ///
  /// \param IsAngled Whether the file name was enclosed in angle brackets;
  /// otherwise, it was enclosed in quotes.
  ///
  /// \param File The actual file that may be included by this embed directive.
  ///
  /// \param Params The parameters used by the directive.
  void EmbedDirective(SourceLocation HashLoc, StringRef FileName, bool IsAngled,
                      OptionalFileEntryRef File,
                      const LexEmbedParametersResult &Params) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Callback invoked whenever an inclusion directive of
  /// any kind (\c \#include, \c \#import, etc.) has been processed, regardless
  /// of whether the inclusion will actually result in an inclusion.
  ///
  /// \param HashLoc The location of the '#' that starts the inclusion
  /// directive.
  ///
  /// \param IncludeTok The token that indicates the kind of inclusion
  /// directive, e.g., 'include' or 'import'.
  ///
  /// \param FileName The name of the file being included, as written in the
  /// source code.
  ///
  /// \param IsAngled Whether the file name was enclosed in angle brackets;
  /// otherwise, it was enclosed in quotes.
  ///
  /// \param FilenameRange The character range of the quotes or angle brackets
  /// for the written file name.
  ///
  /// \param File The actual file that may be included by this inclusion
  /// directive.
  ///
  /// \param SearchPath Contains the search path which was used to find the file
  /// in the file system. If the file was found via an absolute include path,
  /// SearchPath will be empty. For framework includes, the SearchPath and
  /// RelativePath will be split up. For example, if an include of "Some/Some.h"
  /// is found via the framework path
  /// "path/to/Frameworks/Some.framework/Headers/Some.h", SearchPath will be
  /// "path/to/Frameworks/Some.framework/Headers" and RelativePath will be
  /// "Some.h".
  ///
  /// \param RelativePath The path relative to SearchPath, at which the include
  /// file was found. This is equal to FileName except for framework includes.
  ///
  /// \param SuggestedModule The module suggested for this header, if any.
  ///
  /// \param ModuleImported Whether this include was translated into import of
  /// \p SuggestedModule.
  ///
  /// \param FileType The characteristic kind, indicates whether a file or
  /// directory holds normal user code, system code, or system code which is
  /// implicitly 'extern "C"' in C++ mode.
  ///
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *SuggestedModule,
                          bool ModuleImported,
                          SrcMgr::CharacteristicKind FileType) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Callback invoked whenever there was an explicit module-import
  /// syntax.
  ///
  /// \param ImportLoc The location of import directive token.
  ///
  /// \param Path The identifiers (and their locations) of the module
  /// "path", e.g., "std.vector" would be split into "std" and "vector".
  ///
  /// \param Imported The imported module; can be null if importing failed.
  ///
  void moduleImport(SourceLocation ImportLoc, ModuleIdPath Path,
                    const Module *Imported) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Callback invoked when the end of the main file is reached.
  ///
  /// No subsequent callbacks will be made.
  void EndOfMainFile() override { setSeenNoTrivialPPDirective(true); }

  /// Callback invoked when a \#ident or \#sccs directive is read.
  /// \param Loc The location of the directive.
  /// \param str The text of the directive.
  ///
  void Ident(SourceLocation Loc, StringRef str) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when start reading any pragma directive.
  void PragmaDirective(SourceLocation Loc,
                       PragmaIntroducerKind Introducer) override {}

  /// Callback invoked when a \#pragma comment directive is read.
  void PragmaComment(SourceLocation Loc, const IdentifierInfo *Kind,
                     StringRef Str) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma mark comment is read.
  void PragmaMark(SourceLocation Loc, StringRef Trivia) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma detect_mismatch directive is
  /// read.
  void PragmaDetectMismatch(SourceLocation Loc, StringRef Name,
                            StringRef Value) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma clang __debug directive is read.
  /// \param Loc The location of the debug directive.
  /// \param DebugType The identifier following __debug.
  void PragmaDebug(SourceLocation Loc, StringRef DebugType) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma message directive is read.
  /// \param Loc The location of the message directive.
  /// \param Namespace The namespace of the message directive.
  /// \param Kind The type of the message directive.
  /// \param Str The text of the message directive.
  void PragmaMessage(SourceLocation Loc, StringRef Namespace,
                     PragmaMessageKind Kind, StringRef Str) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma gcc diagnostic push directive
  /// is read.
  void PragmaDiagnosticPush(SourceLocation Loc, StringRef Namespace) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma gcc diagnostic pop directive
  /// is read.
  void PragmaDiagnosticPop(SourceLocation Loc, StringRef Namespace) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma gcc diagnostic directive is read.
  void PragmaDiagnostic(SourceLocation Loc, StringRef Namespace,
                        diag::Severity mapping, StringRef Str) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Called when an OpenCL extension is either disabled or
  /// enabled with a pragma.
  void PragmaOpenCLExtension(SourceLocation NameLoc, const IdentifierInfo *Name,
                             SourceLocation StateLoc, unsigned State) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma warning directive is read.
  void PragmaWarning(SourceLocation Loc, PragmaWarningSpecifier WarningSpec,
                     ArrayRef<int> Ids) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma warning(push) directive is read.
  void PragmaWarningPush(SourceLocation Loc, int Level) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma warning(pop) directive is read.
  void PragmaWarningPop(SourceLocation Loc) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma execution_character_set(push) directive
  /// is read.
  void PragmaExecCharsetPush(SourceLocation Loc, StringRef Str) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma execution_character_set(pop) directive
  /// is read.
  void PragmaExecCharsetPop(SourceLocation Loc) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma clang assume_nonnull begin directive
  /// is read.
  void PragmaAssumeNonNullBegin(SourceLocation Loc) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Callback invoked when a \#pragma clang assume_nonnull end directive
  /// is read.
  void PragmaAssumeNonNullEnd(SourceLocation Loc) override {
    setSeenNoTrivialPPDirective(false);
  }

  /// Called by Preprocessor::HandleMacroExpandedIdentifier when a
  /// macro invocation is found.
  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override;

  /// Hook called whenever a macro definition is seen.
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Hook called whenever a macro \#undef is seen.
  /// \param MacroNameTok The active Token
  /// \param MD A MacroDefinition for the named macro.
  /// \param Undef New MacroDirective if the macro was defined, null otherwise.
  ///
  /// MD is released immediately following this callback.
  void MacroUndefined(const Token &MacroNameTok, const MacroDefinition &MD,
                      const MacroDirective *Undef) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Hook called whenever the 'defined' operator is seen.
  /// \param MD The MacroDirective if the name was a macro, null otherwise.
  void Defined(const Token &MacroNameTok, const MacroDefinition &MD,
               SourceRange Range) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Hook called whenever an \#if is seen.
  /// \param Loc the source location of the directive.
  /// \param ConditionRange The SourceRange of the expression being tested.
  /// \param ConditionValue The evaluated value of the condition.
  ///
  // FIXME: better to pass in a list (or tree!) of Tokens.
  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind ConditionValue) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Hook called whenever an \#elif is seen.
  /// \param Loc the source location of the directive.
  /// \param ConditionRange The SourceRange of the expression being tested.
  /// \param ConditionValue The evaluated value of the condition.
  /// \param IfLoc the source location of the \#if/\#ifdef/\#ifndef directive.
  // FIXME: better to pass in a list (or tree!) of Tokens.
  void Elif(SourceLocation Loc, SourceRange ConditionRange,
            ConditionValueKind ConditionValue, SourceLocation IfLoc) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Hook called whenever an \#ifdef is seen.
  /// \param Loc the source location of the directive.
  /// \param MacroNameTok Information on the token being tested.
  /// \param MD The MacroDefinition if the name was a macro, null otherwise.
  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MD) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Hook called whenever an \#elifdef branch is taken.
  /// \param Loc the source location of the directive.
  /// \param MacroNameTok Information on the token being tested.
  /// \param MD The MacroDefinition if the name was a macro, null otherwise.
  void Elifdef(SourceLocation Loc, const Token &MacroNameTok,
               const MacroDefinition &MD) override {
    setSeenNoTrivialPPDirective(true);
  }
  /// Hook called whenever an \#elifdef is skipped.
  /// \param Loc the source location of the directive.
  /// \param ConditionRange The SourceRange of the expression being tested.
  /// \param IfLoc the source location of the \#if/\#ifdef/\#ifndef directive.
  // FIXME: better to pass in a list (or tree!) of Tokens.
  void Elifdef(SourceLocation Loc, SourceRange ConditionRange,
               SourceLocation IfLoc) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Hook called whenever an \#ifndef is seen.
  /// \param Loc the source location of the directive.
  /// \param MacroNameTok Information on the token being tested.
  /// \param MD The MacroDefiniton if the name was a macro, null otherwise.
  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Hook called whenever an \#elifndef branch is taken.
  /// \param Loc the source location of the directive.
  /// \param MacroNameTok Information on the token being tested.
  /// \param MD The MacroDefinition if the name was a macro, null otherwise.
  void Elifndef(SourceLocation Loc, const Token &MacroNameTok,
                const MacroDefinition &MD) override {
    setSeenNoTrivialPPDirective(true);
  }
  /// Hook called whenever an \#elifndef is skipped.
  /// \param Loc the source location of the directive.
  /// \param ConditionRange The SourceRange of the expression being tested.
  /// \param IfLoc the source location of the \#if/\#ifdef/\#ifndef directive.
  // FIXME: better to pass in a list (or tree!) of Tokens.
  void Elifndef(SourceLocation Loc, SourceRange ConditionRange,
                SourceLocation IfLoc) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Hook called whenever an \#else is seen.
  /// \param Loc the source location of the directive.
  /// \param IfLoc the source location of the \#if/\#ifdef/\#ifndef directive.
  void Else(SourceLocation Loc, SourceLocation IfLoc) override {
    setSeenNoTrivialPPDirective(true);
  }

  /// Hook called whenever an \#endif is seen.
  /// \param Loc the source location of the directive.
  /// \param IfLoc the source location of the \#if/\#ifdef/\#ifndef directive.
  void Endif(SourceLocation Loc, SourceLocation IfLoc) override {
    setSeenNoTrivialPPDirective(true);
  }
};

} // namespace clang

#endif // LLVM_CLANG_LEX_TRIVIAL_DIRECTIVE_TRACER_H
