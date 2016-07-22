//===--- DiagnosticRenderer.h - Diagnostic Pretty-Printing ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility class that provides support for pretty-printing of
// diagnostics. It is used to implement the different code paths which require
// such functionality in a consistent way.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_DIAGNOSTICRENDERER_H
#define LLVM_CLANG_FRONTEND_DIAGNOSTICRENDERER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/PointerUnion.h"

namespace clang {

class DiagnosticOptions;
class LangOptions;
class SourceManager;

typedef llvm::PointerUnion<const Diagnostic *,
                           const StoredDiagnostic *> DiagOrStoredDiag;
  
/// \brief Class to encapsulate the logic for formatting a diagnostic message.
///
/// Actual "printing" logic is implemented by subclasses.
///
/// This class provides an interface for building and emitting
/// diagnostic, including all of the macro backtraces, caret diagnostics, FixIt
/// Hints, and code snippets. In the presence of macros this involves
/// a recursive process, synthesizing notes for each macro expansion.
///
/// A brief worklist:
/// FIXME: Sink the recursive printing of template instantiations into this
/// class.
class DiagnosticRenderer {
protected:
  const LangOptions &LangOpts;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  
  /// \brief The location of the previous diagnostic if known.
  ///
  /// This will be invalid in cases where there is no (known) previous
  /// diagnostic location, or that location itself is invalid or comes from
  /// a different source manager than SM.
  SourceLocation LastLoc;
  
  /// \brief The location of the last include whose stack was printed if known.
  ///
  /// Same restriction as LastLoc essentially, but tracking include stack
  /// root locations rather than diagnostic locations.
  SourceLocation LastIncludeLoc;
  
  /// \brief The level of the last diagnostic emitted.
  ///
  /// The level of the last diagnostic emitted. Used to detect level changes
  /// which change the amount of information displayed.
  DiagnosticsEngine::Level LastLevel;

  DiagnosticRenderer(const LangOptions &LangOpts,
                     DiagnosticOptions *DiagOpts);
  
  virtual ~DiagnosticRenderer();
  
  virtual void emitDiagnosticMessage(SourceLocation Loc, PresumedLoc PLoc,
                                     DiagnosticsEngine::Level Level,
                                     StringRef Message,
                                     ArrayRef<CharSourceRange> Ranges,
                                     const SourceManager *SM,
                                     DiagOrStoredDiag Info) = 0;
  
  virtual void emitDiagnosticLoc(SourceLocation Loc, PresumedLoc PLoc,
                                 DiagnosticsEngine::Level Level,
                                 ArrayRef<CharSourceRange> Ranges,
                                 const SourceManager &SM) = 0;

  virtual void emitCodeContext(SourceLocation Loc,
                               DiagnosticsEngine::Level Level,
                               SmallVectorImpl<CharSourceRange>& Ranges,
                               ArrayRef<FixItHint> Hints,
                               const SourceManager &SM) = 0;
  
  virtual void emitIncludeLocation(SourceLocation Loc, PresumedLoc PLoc,
                                   const SourceManager &SM) = 0;
  virtual void emitImportLocation(SourceLocation Loc, PresumedLoc PLoc,
                                  StringRef ModuleName,
                                  const SourceManager &SM) = 0;
  virtual void emitBuildingModuleLocation(SourceLocation Loc, PresumedLoc PLoc,
                                          StringRef ModuleName,
                                          const SourceManager &SM) = 0;

  virtual void beginDiagnostic(DiagOrStoredDiag D,
                               DiagnosticsEngine::Level Level) {}
  virtual void endDiagnostic(DiagOrStoredDiag D,
                             DiagnosticsEngine::Level Level) {}

  
private:
  void emitBasicNote(StringRef Message);
  void emitIncludeStack(SourceLocation Loc, PresumedLoc PLoc,
                        DiagnosticsEngine::Level Level, const SourceManager &SM);
  void emitIncludeStackRecursively(SourceLocation Loc, const SourceManager &SM);
  void emitImportStack(SourceLocation Loc, const SourceManager &SM);
  void emitImportStackRecursively(SourceLocation Loc, StringRef ModuleName,
                                  const SourceManager &SM);
  void emitModuleBuildStack(const SourceManager &SM);
  void emitCaret(SourceLocation Loc, DiagnosticsEngine::Level Level,
                 ArrayRef<CharSourceRange> Ranges, ArrayRef<FixItHint> Hints,
                 const SourceManager &SM);
  void emitSingleMacroExpansion(SourceLocation Loc,
                                DiagnosticsEngine::Level Level,
                                ArrayRef<CharSourceRange> Ranges,
                                const SourceManager &SM);
  void emitMacroExpansions(SourceLocation Loc,
                           DiagnosticsEngine::Level Level,
                           ArrayRef<CharSourceRange> Ranges,
                           ArrayRef<FixItHint> Hints,
                           const SourceManager &SM);
public:
  /// \brief Emit a diagnostic.
  ///
  /// This is the primary entry point for emitting diagnostic messages.
  /// It handles formatting and rendering the message as well as any ancillary
  /// information needed based on macros whose expansions impact the
  /// diagnostic.
  ///
  /// \param Loc The location for this caret.
  /// \param Level The level of the diagnostic to be emitted.
  /// \param Message The diagnostic message to emit.
  /// \param Ranges The underlined ranges for this code snippet.
  /// \param FixItHints The FixIt hints active for this diagnostic.
  /// \param SM The SourceManager; will be null if the diagnostic came from the
  ///        frontend, thus \p Loc will be invalid.
  void emitDiagnostic(SourceLocation Loc, DiagnosticsEngine::Level Level,
                      StringRef Message, ArrayRef<CharSourceRange> Ranges,
                      ArrayRef<FixItHint> FixItHints,
                      const SourceManager *SM,
                      DiagOrStoredDiag D = (Diagnostic *)nullptr);

  void emitStoredDiagnostic(StoredDiagnostic &Diag);
};
  
/// Subclass of DiagnosticRender that turns all subdiagostics into explicit
/// notes.  It is up to subclasses to further define the behavior.
class DiagnosticNoteRenderer : public DiagnosticRenderer {
public:
  DiagnosticNoteRenderer(const LangOptions &LangOpts,
                         DiagnosticOptions *DiagOpts)
    : DiagnosticRenderer(LangOpts, DiagOpts) {}

  ~DiagnosticNoteRenderer() override;

  void emitIncludeLocation(SourceLocation Loc, PresumedLoc PLoc,
                           const SourceManager &SM) override;

  void emitImportLocation(SourceLocation Loc, PresumedLoc PLoc,
                          StringRef ModuleName,
                          const SourceManager &SM) override;

  void emitBuildingModuleLocation(SourceLocation Loc, PresumedLoc PLoc,
                                  StringRef ModuleName,
                                  const SourceManager &SM) override;

  virtual void emitNote(SourceLocation Loc, StringRef Message,
                        const SourceManager *SM) = 0;
};
} // end clang namespace
#endif
