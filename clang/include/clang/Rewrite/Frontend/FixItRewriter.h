//===--- FixItRewriter.h - Fix-It Rewriter Diagnostic Client ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a diagnostic client adaptor that performs rewrites as
// suggested by code modification hints attached to diagnostics. It
// then forwards any diagnostics to the adapted diagnostic client.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_REWRITE_FIX_IT_REWRITER_H
#define LLVM_CLANG_REWRITE_FIX_IT_REWRITER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Edit/EditedSource.h"

namespace clang {

class SourceManager;
class FileEntry;

class FixItOptions {
public:
  FixItOptions() : FixWhatYouCan(false),
                   FixOnlyWarnings(false), Silent(false) { }

  virtual ~FixItOptions();

  /// \brief This file is about to be rewritten. Return the name of the file
  /// that is okay to write to.
  ///
  /// \param fd out parameter for file descriptor. After the call it may be set
  /// to an open file descriptor for the returned filename, or it will be -1
  /// otherwise.
  ///
  virtual std::string RewriteFilename(const std::string &Filename, int &fd) = 0;

  /// \brief Whether to abort fixing a file when not all errors could be fixed.
  bool FixWhatYouCan;

  /// \brief Whether to only fix warnings and not errors.
  bool FixOnlyWarnings;

  /// \brief If true, only pass the diagnostic to the actual diagnostic consumer
  /// if it is an error or a fixit was applied as part of the diagnostic.
  /// It basically silences warnings without accompanying fixits.
  bool Silent;
};

class FixItRewriter : public DiagnosticConsumer {
  /// \brief The diagnostics machinery.
  DiagnosticsEngine &Diags;

  edit::EditedSource Editor;

  /// \brief The rewriter used to perform the various code
  /// modifications.
  Rewriter Rewrite;

  /// \brief The diagnostic client that performs the actual formatting
  /// of error messages.
  DiagnosticConsumer *Client;
  bool OwnsClient;

  /// \brief Turn an input path into an output path. NULL implies overwriting
  /// the original.
  FixItOptions *FixItOpts;

  /// \brief The number of rewriter failures.
  unsigned NumFailures;

  /// \brief Whether the previous diagnostic was not passed to the consumer.
  bool PrevDiagSilenced;

public:
  typedef Rewriter::buffer_iterator iterator;

  /// \brief Initialize a new fix-it rewriter.
  FixItRewriter(DiagnosticsEngine &Diags, SourceManager &SourceMgr,
                const LangOptions &LangOpts, FixItOptions *FixItOpts);

  /// \brief Destroy the fix-it rewriter.
  ~FixItRewriter();

  /// \brief Check whether there are modifications for a given file.
  bool IsModified(FileID ID) const {
    return Rewrite.getRewriteBufferFor(ID) != NULL;
  }

  // Iteration over files with changes.
  iterator buffer_begin() { return Rewrite.buffer_begin(); }
  iterator buffer_end() { return Rewrite.buffer_end(); }

  /// \brief Write a single modified source file.
  ///
  /// \returns true if there was an error, false otherwise.
  bool WriteFixedFile(FileID ID, raw_ostream &OS);

  /// \brief Write the modified source files.
  ///
  /// \returns true if there was an error, false otherwise.
  bool WriteFixedFiles(
         std::vector<std::pair<std::string, std::string> > *RewrittenFiles = 0);

  /// IncludeInDiagnosticCounts - This method (whose default implementation
  /// returns true) indicates whether the diagnostics handled by this
  /// DiagnosticConsumer should be included in the number of diagnostics
  /// reported by DiagnosticsEngine.
  virtual bool IncludeInDiagnosticCounts() const;

  /// HandleDiagnostic - Handle this diagnostic, reporting it to the user or
  /// capturing it to a log as needed.
  virtual void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                const Diagnostic &Info);

  /// \brief Emit a diagnostic via the adapted diagnostic client.
  void Diag(SourceLocation Loc, unsigned DiagID);
  
  DiagnosticConsumer *clone(DiagnosticsEngine &Diags) const;
};

}

#endif // LLVM_CLANG_REWRITE_FIX_IT_REWRITER_H
