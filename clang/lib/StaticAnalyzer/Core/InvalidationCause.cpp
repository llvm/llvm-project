//===- InvalidationCause.cpp - Cause of a region invalidation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/InvalidationCause.h"
#include "clang/AST/Stmt.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

void InvalidationCause::anchor() {}
void UnmodeledCall::anchor() {}
void ConservativeEvalCall::anchor() {}
void PartiallyModeledCall::anchor() {}
void UnmodeledStmt::anchor() {}
void UnmodeledExpr::anchor() {}
void LoopWidening::anchor() {}

LLVM_DUMP_METHOD void InvalidationCause::dump() const {
  dumpToStream(llvm::errs());
}

raw_ostream &ento::operator<<(raw_ostream &OS, const InvalidationCause &C) {
  C.dumpToStream(OS);
  return OS;
}

void ConservativeEvalCall::dumpToStream(raw_ostream &OS) const {
  OS << "conservative-call";
}

void PartiallyModeledCall::dumpToStream(raw_ostream &OS) const {
  OS << "partial-call";
}

void UnmodeledExpr::dumpToStream(raw_ostream &OS) const {
  OS << "unmodeled-expr " << S->getStmtClassName();
}

void LoopWidening::dumpToStream(raw_ostream &OS) const {
  OS << "loop-widening";
}