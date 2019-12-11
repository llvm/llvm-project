//=== Iterator.cpp - Common functions for iterator checkers. -------*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines common functions to be used by the itertor checkers .
//
//===----------------------------------------------------------------------===//

#include "Iterator.h"

namespace clang {
namespace ento {
namespace iterator {

bool isIteratorType(const QualType &Type) {
  if (Type->isPointerType())
    return true;

  const auto *CRD = Type->getUnqualifiedDesugaredType()->getAsCXXRecordDecl();
  return isIterator(CRD);
}

bool isIterator(const CXXRecordDecl *CRD) {
  if (!CRD)
    return false;

  const auto Name = CRD->getName();
  if (!(Name.endswith_lower("iterator") || Name.endswith_lower("iter") ||
        Name.endswith_lower("it")))
    return false;

  bool HasCopyCtor = false, HasCopyAssign = true, HasDtor = false,
       HasPreIncrOp = false, HasPostIncrOp = false, HasDerefOp = false;
  for (const auto *Method : CRD->methods()) {
    if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(Method)) {
      if (Ctor->isCopyConstructor()) {
        HasCopyCtor = !Ctor->isDeleted() && Ctor->getAccess() == AS_public;
      }
      continue;
    }
    if (const auto *Dtor = dyn_cast<CXXDestructorDecl>(Method)) {
      HasDtor = !Dtor->isDeleted() && Dtor->getAccess() == AS_public;
      continue;
    }
    if (Method->isCopyAssignmentOperator()) {
      HasCopyAssign = !Method->isDeleted() && Method->getAccess() == AS_public;
      continue;
    }
    if (!Method->isOverloadedOperator())
      continue;
    const auto OPK = Method->getOverloadedOperator();
    if (OPK == OO_PlusPlus) {
      HasPreIncrOp = HasPreIncrOp || (Method->getNumParams() == 0);
      HasPostIncrOp = HasPostIncrOp || (Method->getNumParams() == 1);
      continue;
    }
    if (OPK == OO_Star) {
      HasDerefOp = (Method->getNumParams() == 0);
      continue;
    }
  }

  return HasCopyCtor && HasCopyAssign && HasDtor && HasPreIncrOp &&
         HasPostIncrOp && HasDerefOp;
}

bool isComparisonOperator(OverloadedOperatorKind OK) {
  return OK == OO_EqualEqual || OK == OO_ExclaimEqual || OK == OO_Less ||
         OK == OO_LessEqual || OK == OO_Greater || OK == OO_GreaterEqual;
}

bool isInsertCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() < 2 || Func->getNumParams() > 3)
    return false;
  if (!isIteratorType(Func->getParamDecl(0)->getType()))
    return false;
  return IdInfo->getName() == "insert";
}

bool isEmplaceCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() < 2)
    return false;
  if (!isIteratorType(Func->getParamDecl(0)->getType()))
    return false;
  return IdInfo->getName() == "emplace";
}

bool isEraseCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() < 1 || Func->getNumParams() > 2)
    return false;
  if (!isIteratorType(Func->getParamDecl(0)->getType()))
    return false;
  if (Func->getNumParams() == 2 &&
      !isIteratorType(Func->getParamDecl(1)->getType()))
    return false;
  return IdInfo->getName() == "erase";
}

bool isEraseAfterCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() < 1 || Func->getNumParams() > 2)
    return false;
  if (!isIteratorType(Func->getParamDecl(0)->getType()))
    return false;
  if (Func->getNumParams() == 2 &&
      !isIteratorType(Func->getParamDecl(1)->getType()))
    return false;
  return IdInfo->getName() == "erase_after";
}

bool isAccessOperator(OverloadedOperatorKind OK) {
  return isDereferenceOperator(OK) || isIncrementOperator(OK) ||
         isDecrementOperator(OK) || isRandomIncrOrDecrOperator(OK);
}

bool isDereferenceOperator(OverloadedOperatorKind OK) {
  return OK == OO_Star || OK == OO_Arrow || OK == OO_ArrowStar ||
         OK == OO_Subscript;
}

bool isIncrementOperator(OverloadedOperatorKind OK) {
  return OK == OO_PlusPlus;
}

bool isDecrementOperator(OverloadedOperatorKind OK) {
  return OK == OO_MinusMinus;
}

bool isRandomIncrOrDecrOperator(OverloadedOperatorKind OK) {
  return OK == OO_Plus || OK == OO_PlusEqual || OK == OO_Minus ||
         OK == OO_MinusEqual;
}

const ContainerData *getContainerData(ProgramStateRef State,
                                      const MemRegion *Cont) {
  return State->get<ContainerMap>(Cont);
}

const IteratorPosition *getIteratorPosition(ProgramStateRef State,
                                            const SVal &Val) {
  if (auto Reg = Val.getAsRegion()) {
    Reg = Reg->getMostDerivedObjectRegion();
    return State->get<IteratorRegionMap>(Reg);
  } else if (const auto Sym = Val.getAsSymbol()) {
    return State->get<IteratorSymbolMap>(Sym);
  } else if (const auto LCVal = Val.getAs<nonloc::LazyCompoundVal>()) {
    return State->get<IteratorRegionMap>(LCVal->getRegion());
  }
  return nullptr;
}

ProgramStateRef setIteratorPosition(ProgramStateRef State, const SVal &Val,
                                    const IteratorPosition &Pos) {
  if (auto Reg = Val.getAsRegion()) {
    Reg = Reg->getMostDerivedObjectRegion();
    return State->set<IteratorRegionMap>(Reg, Pos);
  } else if (const auto Sym = Val.getAsSymbol()) {
    return State->set<IteratorSymbolMap>(Sym, Pos);
  } else if (const auto LCVal = Val.getAs<nonloc::LazyCompoundVal>()) {
    return State->set<IteratorRegionMap>(LCVal->getRegion(), Pos);
  }
  return nullptr;
}

ProgramStateRef advancePosition(ProgramStateRef State, const SVal &Iter,
                                OverloadedOperatorKind Op,
                                const SVal &Distance) {
  const auto *Pos = getIteratorPosition(State, Iter);
  if (!Pos)
    return nullptr;

  auto &SymMgr = State->getStateManager().getSymbolManager();
  auto &SVB = State->getStateManager().getSValBuilder();

  assert ((Op == OO_Plus || Op == OO_PlusEqual ||
           Op == OO_Minus || Op == OO_MinusEqual) &&
          "Advance operator must be one of +, -, += and -=.");
  auto BinOp = (Op == OO_Plus || Op == OO_PlusEqual) ? BO_Add : BO_Sub;
  if (const auto IntDist = Distance.getAs<nonloc::ConcreteInt>()) {
    // For concrete integers we can calculate the new position
    const auto NewPos =
      Pos->setTo(SVB.evalBinOp(State, BinOp,
                               nonloc::SymbolVal(Pos->getOffset()),
                               *IntDist, SymMgr.getType(Pos->getOffset()))
                 .getAsSymbol());
    return setIteratorPosition(State, Iter, NewPos);
  }

  return nullptr;
}

bool compare(ProgramStateRef State, SymbolRef Sym1, SymbolRef Sym2,
             BinaryOperator::Opcode Opc) {
  return compare(State, nonloc::SymbolVal(Sym1), nonloc::SymbolVal(Sym2), Opc);
}

bool compare(ProgramStateRef State, NonLoc NL1, NonLoc NL2,
             BinaryOperator::Opcode Opc) {
  auto &SVB = State->getStateManager().getSValBuilder();

  const auto comparison =
    SVB.evalBinOp(State, Opc, NL1, NL2, SVB.getConditionType());

  assert(comparison.getAs<DefinedSVal>() &&
    "Symbol comparison must be a `DefinedSVal`");

  return !State->assume(comparison.castAs<DefinedSVal>(), false);
}

} // namespace iterator
} // namespace ento
} // namespace clang
