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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstddef>
#include <stack>
#include <vector>

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

// Avoids recursion to avoid stack overflows from very large formulas.
void serializeFormula(const Formula &F, llvm::raw_ostream &OS) {
  std::stack<const Formula *> WorkList;
  WorkList.push(&F);
  while (!WorkList.empty()) {
    const Formula *Current = WorkList.top();
    WorkList.pop();
    switch (Formula::numOperands(Current->kind())) {
    case 0:
      switch (Current->kind()) {
      case Formula::AtomRef:
        OS << Current->getAtom();
        break;
      case Formula::Literal:
        OS << (Current->literal() ? 'T' : 'F');
        break;
      default:
        llvm_unreachable("unhandled formula kind");
      }
      break;
    case 1:
      OS << compactSigil(Current->kind());
      WorkList.push(Current->operands()[0]);
      break;
    case 2:
      OS << compactSigil(Current->kind());
      WorkList.push(Current->operands()[1]);
      WorkList.push(Current->operands()[0]);
      break;
    default:
      llvm_unreachable("unhandled formula arity");
    }
  }
}

struct Operation {
  Operation(Formula::Kind Kind) : Kind(Kind) {}
  const Formula::Kind Kind;
  const unsigned ExpectedNumOperands = Formula::numOperands(Kind);
  std::vector<const Formula *> Operands;
};

// Avoids recursion to avoid stack overflows from very large formulas.
static llvm::Expected<const Formula *>
parseFormulaInternal(llvm::StringRef &Str, Arena &A,
                     llvm::DenseMap<unsigned, Atom> &AtomMap) {
  std::stack<Operation> ActiveOperations;

  while (true) {
    if (ActiveOperations.empty() ||
        ActiveOperations.top().ExpectedNumOperands >
            ActiveOperations.top().Operands.size()) {
      if (Str.empty()) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "unexpected end of input");
      }
      char Prefix = Str[0];
      Str = Str.drop_front();

      switch (Prefix) {
      // Terminals
      case 'T':
      case 'F':
      case 'V': {
        const Formula *TerminalFormula;
        switch (Prefix) {
        case 'T':
          TerminalFormula = &A.makeLiteral(true);
          break;
        case 'F':
          TerminalFormula = &A.makeLiteral(false);
          break;
        case 'V': {
          unsigned AtomID;
          if (Str.consumeInteger(10, AtomID))
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "expected atom id");
          auto [It, Inserted] = AtomMap.try_emplace(AtomID, Atom());
          if (Inserted)
            It->second = A.makeAtom();
          TerminalFormula = &A.makeAtomRef(It->second);
          break;
        }
        default:
          llvm_unreachable("unexpected terminal character");
        }
        if (ActiveOperations.empty()) {
          return TerminalFormula;
        }
        Operation *Op = &ActiveOperations.top();
        Op->Operands.push_back(TerminalFormula);
      } break;
      case '!':
        ActiveOperations.emplace(Formula::Kind::Not);
        break;
      case '&':
        ActiveOperations.emplace(Formula::Kind::And);
        break;
      case '|':
        ActiveOperations.emplace(Formula::Kind::Or);
        break;
      case '>':
        ActiveOperations.emplace(Formula::Kind::Implies);
        break;
      case '=':
        ActiveOperations.emplace(Formula::Kind::Equal);
        break;
      default:
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "unexpected prefix character: %c",
                                       Prefix);
      }
    } else if (!ActiveOperations.empty() &&
               ActiveOperations.top().ExpectedNumOperands ==
                   ActiveOperations.top().Operands.size()) {
      Operation *Op = &ActiveOperations.top();
      const Formula *OpFormula = nullptr;
      switch (Op->Kind) {
      case Formula::Kind::Not:
        OpFormula = &A.makeNot(*Op->Operands[0]);
        break;
      case Formula::Kind::And:
        OpFormula = &A.makeAnd(*Op->Operands[0], *Op->Operands[1]);
        break;
      case Formula::Kind::Or:
        OpFormula = &A.makeOr(*Op->Operands[0], *Op->Operands[1]);
        break;
      case Formula::Kind::Implies:
        OpFormula = &A.makeImplies(*Op->Operands[0], *Op->Operands[1]);
        break;
      case Formula::Kind::Equal:
        OpFormula = &A.makeEquals(*Op->Operands[0], *Op->Operands[1]);
        break;
      default:
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "only unary and binary operations are "
                                       "expected, but got Formula::Kind %u",
                                       Op->Kind);
      }
      ActiveOperations.pop();
      if (ActiveOperations.empty())
        return OpFormula;
      Op = &ActiveOperations.top();
      Op->Operands.push_back(OpFormula);
    } else {
      llvm_unreachable(
          "we should never have added more operands than expected");
    }
  }
}

llvm::Expected<const Formula *>
parseFormula(llvm::StringRef Str, Arena &A,
             llvm::DenseMap<unsigned, Atom> &AtomMap) {
  size_t OriginalSize = Str.size();
  llvm::Expected<const Formula *> F = parseFormulaInternal(Str, A, AtomMap);
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
