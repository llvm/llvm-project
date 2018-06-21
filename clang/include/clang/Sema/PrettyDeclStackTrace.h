//===- PrettyDeclStackTrace.h - Stack trace for decl processing -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an llvm::PrettyStackTraceEntry object for showing
// that a particular declaration was being processed when a crash
// occurred.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_PRETTYDECLSTACKTRACE_H
#define LLVM_CLANG_SEMA_PRETTYDECLSTACKTRACE_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/PrettyStackTrace.h"

namespace clang {

class Decl;
class Sema;
class SourceManager;

/// PrettyDeclStackTraceEntry - If a crash occurs in the parser while
/// parsing something related to a declaration, include that
/// declaration in the stack trace.
class PrettyDeclStackTraceEntry : public llvm::PrettyStackTraceEntry {
  Sema &S;
  Decl *TheDecl;
  SourceLocation Loc;
  const char *Message;

public:
  PrettyDeclStackTraceEntry(Sema &S, Decl *D, SourceLocation Loc,
                            const char *Msg)
    : S(S), TheDecl(D), Loc(Loc), Message(Msg) {}

  void print(raw_ostream &OS) const override;
};

}

#endif
