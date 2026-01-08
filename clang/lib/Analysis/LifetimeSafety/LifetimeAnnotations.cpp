//===- LifetimeAnnotations.cpp -  -*--------------- C++------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"

namespace clang::lifetimes {

const FunctionDecl *
getDeclWithMergedLifetimeBoundAttrs(const FunctionDecl *FD) {
  return FD != nullptr ? FD->getMostRecentDecl() : nullptr;
}

const CXXMethodDecl *
getDeclWithMergedLifetimeBoundAttrs(const CXXMethodDecl *CMD) {
  const FunctionDecl *FD = CMD;
  return cast_if_present<CXXMethodDecl>(
      getDeclWithMergedLifetimeBoundAttrs(FD));
}

bool isNormalAssignmentOperator(const FunctionDecl *FD) {
  OverloadedOperatorKind OO = FD->getDeclName().getCXXOverloadedOperator();
  bool IsAssignment = OO == OO_Equal || isCompoundAssignmentOperator(OO);
  if (!IsAssignment)
    return false;
  QualType RetT = FD->getReturnType();
  if (!RetT->isLValueReferenceType())
    return false;
  ASTContext &Ctx = FD->getASTContext();
  QualType LHST;
  auto *MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD && MD->isCXXInstanceMember())
    LHST = Ctx.getLValueReferenceType(MD->getFunctionObjectParameterType());
  else
    LHST = FD->getParamDecl(0)->getType();
  return Ctx.hasSameType(RetT, LHST);
}

bool isAssignmentOperatorLifetimeBound(const CXXMethodDecl *CMD) {
  CMD = getDeclWithMergedLifetimeBoundAttrs(CMD);
  return CMD && isNormalAssignmentOperator(CMD) && CMD->param_size() == 1 &&
         CMD->getParamDecl(0)->hasAttr<clang::LifetimeBoundAttr>();
}

bool implicitObjectParamIsLifetimeBound(const FunctionDecl *FD) {
  FD = getDeclWithMergedLifetimeBoundAttrs(FD);
  const TypeSourceInfo *TSI = FD->getTypeSourceInfo();
  if (!TSI)
    return false;
  // Don't declare this variable in the second operand of the for-statement;
  // GCC miscompiles that by ending its lifetime before evaluating the
  // third operand. See gcc.gnu.org/PR86769.
  AttributedTypeLoc ATL;
  for (TypeLoc TL = TSI->getTypeLoc();
       (ATL = TL.getAsAdjusted<AttributedTypeLoc>());
       TL = ATL.getModifiedLoc()) {
    if (ATL.getAttrAs<clang::LifetimeBoundAttr>())
      return true;
  }

  return isNormalAssignmentOperator(FD);
}

bool isInStlNamespace(const Decl *D) {
  const DeclContext *DC = D->getDeclContext();
  if (!DC)
    return false;
  if (const auto *ND = dyn_cast<NamespaceDecl>(DC))
    if (const IdentifierInfo *II = ND->getIdentifier()) {
      StringRef Name = II->getName();
      if (Name.size() >= 2 && Name.front() == '_' &&
          (Name[1] == '_' || isUppercase(Name[1])))
        return true;
    }
  return DC->isStdNamespace();
}

bool isPointerLikeType(QualType QT) {
  return isGslPointerType(QT) || QT->isPointerType() || QT->isNullPtrType();
}

bool shouldTrackImplicitObjectArg(const CXXMethodDecl *Callee) {
  if (auto *Conv = dyn_cast_or_null<CXXConversionDecl>(Callee))
    if (isGslPointerType(Conv->getConversionType()) &&
        Callee->getParent()->hasAttr<OwnerAttr>())
      return true;
  if (!isInStlNamespace(Callee->getParent()))
    return false;
  if (!isGslPointerType(Callee->getFunctionObjectParameterType()) &&
      !isGslOwnerType(Callee->getFunctionObjectParameterType()))
    return false;
  if (isPointerLikeType(Callee->getReturnType())) {
    if (!Callee->getIdentifier())
      return false;
    return llvm::StringSwitch<bool>(Callee->getName())
        .Cases(
            {// Begin and end iterators.
             "begin", "end", "rbegin", "rend", "cbegin", "cend", "crbegin",
             "crend",
             // Inner pointer getters.
             "c_str", "data", "get",
             // Map and set types.
             "find", "equal_range", "lower_bound", "upper_bound"},
            true)
        .Default(false);
  }
  if (Callee->getReturnType()->isReferenceType()) {
    if (!Callee->getIdentifier()) {
      auto OO = Callee->getOverloadedOperator();
      if (!Callee->getParent()->hasAttr<OwnerAttr>())
        return false;
      return OO == OverloadedOperatorKind::OO_Subscript ||
             OO == OverloadedOperatorKind::OO_Star;
    }
    return llvm::StringSwitch<bool>(Callee->getName())
        .Cases({"front", "back", "at", "top", "value"}, true)
        .Default(false);
  }
  return false;
}

bool shouldTrackFirstArgument(const FunctionDecl *FD) {
  if (!FD->getIdentifier() || FD->getNumParams() != 1)
    return false;
  const auto *RD = FD->getParamDecl(0)->getType()->getPointeeCXXRecordDecl();
  if (!FD->isInStdNamespace() || !RD || !RD->isInStdNamespace())
    return false;
  if (!RD->hasAttr<PointerAttr>() && !RD->hasAttr<OwnerAttr>())
    return false;
  if (FD->getReturnType()->isPointerType() ||
      isGslPointerType(FD->getReturnType())) {
    return llvm::StringSwitch<bool>(FD->getName())
        .Cases({"begin", "rbegin", "cbegin", "crbegin"}, true)
        .Cases({"end", "rend", "cend", "crend"}, true)
        .Case("data", true)
        .Default(false);
  }
  if (FD->getReturnType()->isReferenceType()) {
    return llvm::StringSwitch<bool>(FD->getName())
        .Cases({"get", "any_cast"}, true)
        .Default(false);
  }
  return false;
}

template <typename T> static bool isRecordWithAttr(QualType Type) {
  auto *RD = Type->getAsCXXRecordDecl();
  if (!RD)
    return false;
  // Generally, if a primary template class declaration is annotated with an
  // attribute, all its specializations generated from template instantiations
  // should inherit the attribute.
  //
  // However, since lifetime analysis occurs during parsing, we may encounter
  // cases where a full definition of the specialization is not required. In
  // such cases, the specialization declaration remains incomplete and lacks the
  // attribute. Therefore, we fall back to checking the primary template class.
  //
  // Note: it is possible for a specialization declaration to have an attribute
  // even if the primary template does not.
  //
  // FIXME: What if the primary template and explicit specialization
  // declarations have conflicting attributes? We should consider diagnosing
  // this scenario.
  bool Result = RD->hasAttr<T>();

  if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD))
    Result |= CTSD->getSpecializedTemplate()->getTemplatedDecl()->hasAttr<T>();

  return Result;
}

bool isGslPointerType(QualType QT) { return isRecordWithAttr<PointerAttr>(QT); }
bool isGslOwnerType(QualType QT) { return isRecordWithAttr<OwnerAttr>(QT); }

} // namespace clang::lifetimes
