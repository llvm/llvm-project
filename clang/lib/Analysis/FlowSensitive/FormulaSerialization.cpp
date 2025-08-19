//===- FormulaSerialization.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/FormulaSerialization.h"
#include "clang/Analysis/FlowSensitive/Arena.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

namespace clang::dataflow {

// Returns the leading indicator of operation formulas. `AtomRef` and `Literal`
// are handled differently.
static char compactSigil(Formula::Kind K) {
  switch (K) {
  case Formula::AtomRef:
  case Formula::Literal:
    // No sigil.
    return '\0';
  case Formula::Not:
    return '!';
  case Formula::And:
    return '&';
  case Formula::Or:
    return '|';
  case Formula::Implies:
    return '>';
  case Formula::Equal:
    return '=';
  }
  llvm_unreachable("unhandled formula kind");
}

void serializeFormula(const Formula &F, llvm::raw_ostream &OS) {
  switch (Formula::numOperands(F.kind())) {
  case 0:
    switch (F.kind()) {
    case Formula::AtomRef:
      OS << F.getAtom();
      break;
    case Formula::Literal:
      OS << (F.literal() ? 'T' : 'F');
      break;
    default:
      llvm_unreachable("unhandled formula kind");
    }
    break;
  case 1:
    OS << compactSigil(F.kind());
    serializeFormula(*F.operands()[0], OS);
    break;
  case 2:
    OS << compactSigil(F.kind());
    serializeFormula(*F.operands()[0], OS);
    serializeFormula(*F.operands()[1], OS);
    break;
  default:
    llvm_unreachable("unhandled formula arity");
  }
}

static llvm::Expected<const Formula *>
parsePrefix(llvm::StringRef &Str, Arena &A,
            llvm::DenseMap<unsigned, Atom> &AtomMap) {
  if (Str.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unexpected end of input");

  char Prefix = Str[0];
  Str = Str.drop_front();

  switch (Prefix) {
  case 'T':
    return &A.makeLiteral(true);
  case 'F':
    return &A.makeLiteral(false);
  case 'V': {
    unsigned AtomID;
    if (Str.consumeInteger(10, AtomID))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "expected atom id");
    auto [It, Inserted] = AtomMap.try_emplace(AtomID, Atom());
    if (Inserted)
      It->second = A.makeAtom();
    return &A.makeAtomRef(It->second);
  }
  case '!': {
    auto OperandOrErr = parsePrefix(Str, A, AtomMap);
    if (!OperandOrErr)
      return OperandOrErr.takeError();
    return &A.makeNot(**OperandOrErr);
  }
  case '&':
  case '|':
  case '>':
  case '=': {
    auto LeftOrErr = parsePrefix(Str, A, AtomMap);
    if (!LeftOrErr)
      return LeftOrErr.takeError();

    auto RightOrErr = parsePrefix(Str, A, AtomMap);
    if (!RightOrErr)
      return RightOrErr.takeError();

    const Formula &LHS = **LeftOrErr;
    const Formula &RHS = **RightOrErr;

    switch (Prefix) {
    case '&':
      return &A.makeAnd(LHS, RHS);
    case '|':
      return &A.makeOr(LHS, RHS);
    case '>':
      return &A.makeImplies(LHS, RHS);
    case '=':
      return &A.makeEquals(LHS, RHS);
    default:
      llvm_unreachable("unexpected binary op");
    }
  }
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unexpected prefix character: %c", Prefix);
  }
}

llvm::Expected<const Formula *>
parseFormula(llvm::StringRef Str, Arena &A,
             llvm::DenseMap<unsigned, Atom> &AtomMap) {
  size_t OriginalSize = Str.size();
  llvm::Expected<const Formula *> F = parsePrefix(Str, A, AtomMap);
  if (!F)
    return F.takeError();
  if (!Str.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   ("unexpected suffix of length: " +
                                    llvm::Twine(Str.size() - OriginalSize))
                                       .str());
  return F;
}

} // namespace clang::dataflow
