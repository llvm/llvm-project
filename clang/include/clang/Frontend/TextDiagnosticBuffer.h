//===--- TextDiagnosticBuffer.h - Buffer Text Diagnostics -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a concrete diagnostic client, which buffers the diagnostic messages.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_TEXT_DIAGNOSTIC_BUFFER_H_
#define LLVM_CLANG_FRONTEND_TEXT_DIAGNOSTIC_BUFFER_H_

#include "clang/Basic/Diagnostic.h"
#include <vector>

namespace clang {

class Preprocessor;
class SourceManager;

class TextDiagnosticBuffer : public DiagnosticConsumer {
public:
  typedef std::vector<std::pair<SourceLocation, std::string> > DiagList;
  typedef DiagList::iterator iterator;
  typedef DiagList::const_iterator const_iterator;
private:
  DiagList Errors, Warnings, Notes;
public:
  const_iterator err_begin() const  { return Errors.begin(); }
  const_iterator err_end() const    { return Errors.end(); }

  const_iterator warn_begin() const { return Warnings.begin(); }
  const_iterator warn_end() const   { return Warnings.end(); }

  const_iterator note_begin() const { return Notes.begin(); }
  const_iterator note_end() const   { return Notes.end(); }

  virtual void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                const Diagnostic &Info);

  /// FlushDiagnostics - Flush the buffered diagnostics to an given
  /// diagnostic engine.
  void FlushDiagnostics(DiagnosticsEngine &Diags) const;
  
  virtual DiagnosticConsumer *clone(DiagnosticsEngine &Diags) const;
};

} // end namspace clang

#endif
