//==- ProgramPoint.cpp - Program Points for Path-Sensitive Analysis -*- C++ -*-/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface ProgramPoint, which identifies a
//  distinct location in a function.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/ProgramPoint.h"

using namespace clang;

ProgramPointTag::~ProgramPointTag() {}

ProgramPoint ProgramPoint::getProgramPoint(const Stmt *S, ProgramPoint::Kind K,
                                           const LocationContext *LC,
                                           const ProgramPointTag *tag){
  switch (K) {
    default:
      llvm_unreachable("Unhandled ProgramPoint kind");
    case ProgramPoint::PreStmtKind:
      return PreStmt(S, LC, tag);
    case ProgramPoint::PostStmtKind:
      return PostStmt(S, LC, tag);
    case ProgramPoint::PreLoadKind:
      return PreLoad(S, LC, tag);
    case ProgramPoint::PostLoadKind:
      return PostLoad(S, LC, tag);
    case ProgramPoint::PreStoreKind:
      return PreStore(S, LC, tag);
    case ProgramPoint::PostLValueKind:
      return PostLValue(S, LC, tag);
    case ProgramPoint::PostStmtPurgeDeadSymbolsKind:
      return PostStmtPurgeDeadSymbols(S, LC, tag);
    case ProgramPoint::PreStmtPurgeDeadSymbolsKind:
      return PreStmtPurgeDeadSymbols(S, LC, tag);
  }
}

SimpleProgramPointTag::SimpleProgramPointTag(StringRef description)
  : desc(description) {}

StringRef SimpleProgramPointTag::getTagDescription() const {
  return desc;
}
