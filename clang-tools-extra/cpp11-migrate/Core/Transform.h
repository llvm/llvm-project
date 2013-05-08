//===-- cpp11-migrate/Transform.h - Transform Base Class Def'n --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the definition for the base Transform class from
/// which all transforms must subclass.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORM_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORM_H

#include <string>
#include <vector>

// For RewriterContainer
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"
////


/// \brief Description of the riskiness of actions that can be taken by
/// transforms.
enum RiskLevel {
  /// Transformations that will not change semantics.
  RL_Safe,

  /// Transformations that might change semantics.
  RL_Reasonable,

  /// Transformations that are likely to change semantics.
  RL_Risky
};

// Forward declarations
namespace clang {
namespace tooling {
class CompilationDatabase;
} // namespace tooling
} // namespace clang

/// \brief The key is the path of a file, which is mapped to a
/// buffer with the possibly modified contents of that file.
typedef std::map<std::string, std::string> FileContentsByPath;

/// \brief In \p Results place copies of the buffers resulting from applying
/// all rewrites represented by \p Rewrite.
///
/// \p Results is made up of pairs {filename, buffer contents}. Pairs are
/// simply appended to \p Results.
void collectResults(clang::Rewriter &Rewrite,
                    const FileContentsByPath &InputStates,
                    FileContentsByPath &Results);

/// \brief Class for containing a Rewriter instance and all of
/// its lifetime dependencies.
///
/// Subclasses of Transform using RefactoringTools will need to create
/// Rewriters in order to apply Replacements and get the resulting buffer.
/// Rewriter requires some objects to exist at least as long as it does so this
/// class contains instances of those objects.
///
/// FIXME: These objects should really come from somewhere more global instead
/// of being recreated for every Transform subclass, especially diagnostics.
class RewriterContainer {
public:
  RewriterContainer(clang::FileManager &Files,
                    const FileContentsByPath &InputStates)
    : DiagOpts(new clang::DiagnosticOptions()),
      DiagnosticPrinter(llvm::errs(), DiagOpts.getPtr()),
      Diagnostics(llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(
                    new clang::DiagnosticIDs()),
                  DiagOpts.getPtr(), &DiagnosticPrinter, false),
      Sources(Diagnostics, Files),
      Rewrite(Sources, DefaultLangOptions) {

    // Overwrite source manager's file contents with data from InputStates
    for (FileContentsByPath::const_iterator I = InputStates.begin(),
                                            E = InputStates.end();
         I != E; ++I) {
      Sources.overrideFileContents(Files.getFile(I->first),
                                   llvm::MemoryBuffer::getMemBuffer(I->second));
    }
  }

  clang::Rewriter &getRewriter() { return Rewrite; }

private:
  clang::LangOptions DefaultLangOptions;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts;
  clang::TextDiagnosticPrinter DiagnosticPrinter;
  clang::DiagnosticsEngine Diagnostics;
  clang::SourceManager Sources;
  clang::Rewriter Rewrite;
};

/// \brief Abstract base class for all C++11 migration transforms.
class Transform {
public:
  Transform(llvm::StringRef Name) : Name(Name) {
    Reset();
  }

  virtual ~Transform() {}

  /// \brief Apply a transform to all files listed in \p SourcePaths.
  ///
  /// \p Database must contain information for how to compile all files in \p
  /// SourcePaths. \p InputStates contains the file contents of files in \p
  /// SourcePaths and should take precedence over content of files on disk.
  /// Upon return, \p ResultStates shall contain the result of performing this
  /// transform on the files listed in \p SourcePaths.
  virtual int apply(const FileContentsByPath &InputStates,
                    RiskLevel MaxRiskLevel,
                    const clang::tooling::CompilationDatabase &Database,
                    const std::vector<std::string> &SourcePaths,
                    FileContentsByPath &ResultStates) = 0;

  /// \brief Query if changes were made during the last call to apply().
  bool getChangesMade() const { return AcceptedChanges > 0; }

  /// \brief Query if changes were not made due to conflicts with other changes
  /// made during the last call to apply() or if changes were too risky for the
  /// requested risk level.
  bool getChangesNotMade() const {
    return RejectedChanges > 0 || DeferredChanges > 0;
  }

  /// \brief Query the number of accepted changes.
  unsigned getAcceptedChanges() const { return AcceptedChanges; }
  /// \brief Query the number of changes considered too risky.
  unsigned getRejectedChanges() const { return RejectedChanges; }
  /// \brief Query the number of changes not made because they conflicted with
  /// early changes.
  unsigned getDeferredChanges() const { return DeferredChanges; }

  /// \brief Query transform name.
  llvm::StringRef getName() const { return Name; }

  /// \brief Reset internal state of the transform.
  ///
  /// Useful if calling apply() several times with one instantiation of a
  /// transform.
  void Reset() {
    AcceptedChanges = 0;
    RejectedChanges = 0;
    DeferredChanges = 0;
  }

protected:
  void setAcceptedChanges(unsigned Changes) {
    AcceptedChanges = Changes;
  }
  void setRejectedChanges(unsigned Changes) {
    RejectedChanges = Changes;
  }
  void setDeferredChanges(unsigned Changes) {
    DeferredChanges = Changes;
  }

private:
  const std::string Name;
  unsigned AcceptedChanges;
  unsigned RejectedChanges;
  unsigned DeferredChanges;
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORM_H
