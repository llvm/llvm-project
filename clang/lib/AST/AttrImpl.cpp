//===--- AttrImpl.cpp - Classes for representing attributes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file contains out-of-line methods for Attr classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
using namespace clang;

void LoopHintAttr::printPrettyPragma(raw_ostream &OS,
                                     const PrintingPolicy &Policy) const {
  unsigned SpellingIndex = getAttributeSpellingListIndex();
  // For "#pragma unroll" and "#pragma nounroll" the string "unroll" or
  // "nounroll" is already emitted as the pragma name.
  if (SpellingIndex == Pragma_nounroll ||
      SpellingIndex == Pragma_nounroll_and_jam)
    return;
  else if (SpellingIndex == Pragma_unroll ||
           SpellingIndex == Pragma_unroll_and_jam) {
    OS << ' ' << getValueString(Policy);
    return;
  }

  assert(SpellingIndex == Pragma_clang_loop && "Unexpected spelling");
  OS << ' ' << getOptionName(option) << getValueString(Policy);
}

// Return a string containing the loop hint argument including the
// enclosing parentheses.
std::string LoopHintAttr::getValueString(const PrintingPolicy &Policy) const {
  std::string ValueName;
  llvm::raw_string_ostream OS(ValueName);
  OS << "(";
  if (state == Numeric)
    value->printPretty(OS, nullptr, Policy);
  else if (state == FixedWidth || state == ScalableWidth) {
    if (value) {
      value->printPretty(OS, nullptr, Policy);
      if (state == ScalableWidth)
        OS << ", scalable";
    } else if (state == ScalableWidth)
      OS << "scalable";
    else
      OS << "fixed";
  } else if (state == Enable)
    OS << "enable";
  else if (state == Full)
    OS << "full";
  else if (state == AssumeSafety)
    OS << "assume_safety";
  else
    OS << "disable";
  OS << ")";
  return ValueName;
}

// Return a string suitable for identifying this attribute in diagnostics.
std::string
LoopHintAttr::getDiagnosticName(const PrintingPolicy &Policy) const {
  unsigned SpellingIndex = getAttributeSpellingListIndex();
  if (SpellingIndex == Pragma_nounroll)
    return "#pragma nounroll";
  else if (SpellingIndex == Pragma_unroll)
    return "#pragma unroll" +
           (option == UnrollCount ? getValueString(Policy) : "");
  else if (SpellingIndex == Pragma_nounroll_and_jam)
    return "#pragma nounroll_and_jam";
  else if (SpellingIndex == Pragma_unroll_and_jam)
    return "#pragma unroll_and_jam" +
           (option == UnrollAndJamCount ? getValueString(Policy) : "");

  assert(SpellingIndex == Pragma_clang_loop && "Unexpected spelling");
  return getOptionName(option) + getValueString(Policy);
}

void OMPDeclareSimdDeclAttr::printPrettyPragma(
    raw_ostream &OS, const PrintingPolicy &Policy) const {
  if (getBranchState() != BS_Undefined)
    OS << ' ' << ConvertBranchStateTyToStr(getBranchState());
  if (auto *E = getSimdlen()) {
    OS << " simdlen(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  }
  if (uniforms_size() > 0) {
    OS << " uniform";
    StringRef Sep = "(";
    for (auto *E : uniforms()) {
      OS << Sep;
      E->printPretty(OS, nullptr, Policy);
      Sep = ", ";
    }
    OS << ")";
  }
  alignments_iterator NI = alignments_begin();
  for (auto *E : aligneds()) {
    OS << " aligned(";
    E->printPretty(OS, nullptr, Policy);
    if (*NI) {
      OS << ": ";
      (*NI)->printPretty(OS, nullptr, Policy);
    }
    OS << ")";
    ++NI;
  }
  steps_iterator I = steps_begin();
  modifiers_iterator MI = modifiers_begin();
  for (auto *E : linears()) {
    OS << " linear(";
    if (*MI != OMPC_LINEAR_unknown)
      OS << getOpenMPSimpleClauseTypeName(llvm::omp::Clause::OMPC_linear, *MI)
         << "(";
    E->printPretty(OS, nullptr, Policy);
    if (*MI != OMPC_LINEAR_unknown)
      OS << ")";
    if (*I) {
      OS << ": ";
      (*I)->printPretty(OS, nullptr, Policy);
    }
    OS << ")";
    ++I;
    ++MI;
  }
}

void OMPDeclareTargetDeclAttr::printPrettyPragma(
    raw_ostream &OS, const PrintingPolicy &Policy) const {
  // Use fake syntax because it is for testing and debugging purpose only.
  if (getDevType() != DT_Any)
    OS << " device_type(" << ConvertDevTypeTyToStr(getDevType()) << ")";
  if (getMapType() != MT_To && getMapType() != MT_Enter)
    OS << ' ' << ConvertMapTypeTyToStr(getMapType());
  if (Expr *E = getIndirectExpr()) {
    OS << " indirect(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  } else if (getIndirect()) {
    OS << " indirect";
  }
}

llvm::Optional<OMPDeclareTargetDeclAttr *>
OMPDeclareTargetDeclAttr::getActiveAttr(const ValueDecl *VD) {
  if (!VD->hasAttrs())
    return llvm::None;
  unsigned Level = 0;
  OMPDeclareTargetDeclAttr *FoundAttr = nullptr;
  for (auto *Attr : VD->specific_attrs<OMPDeclareTargetDeclAttr>()) {
    if (Level <= Attr->getLevel()) {
      Level = Attr->getLevel();
      FoundAttr = Attr;
    }
  }
  if (FoundAttr)
    return FoundAttr;
  return llvm::None;
}

llvm::Optional<OMPDeclareTargetDeclAttr::MapTypeTy>
OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(const ValueDecl *VD) {
  llvm::Optional<OMPDeclareTargetDeclAttr *> ActiveAttr = getActiveAttr(VD);
  if (ActiveAttr)
    return ActiveAttr.value()->getMapType();
  return llvm::None;
}

llvm::Optional<OMPDeclareTargetDeclAttr::DevTypeTy>
OMPDeclareTargetDeclAttr::getDeviceType(const ValueDecl *VD) {
  llvm::Optional<OMPDeclareTargetDeclAttr *> ActiveAttr = getActiveAttr(VD);
  if (ActiveAttr)
    return ActiveAttr.value()->getDevType();
  return llvm::None;
}

llvm::Optional<SourceLocation>
OMPDeclareTargetDeclAttr::getLocation(const ValueDecl *VD) {
  llvm::Optional<OMPDeclareTargetDeclAttr *> ActiveAttr = getActiveAttr(VD);
  if (ActiveAttr)
    return ActiveAttr.value()->getRange().getBegin();
  return llvm::None;
}

namespace clang {
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const OMPTraitInfo &TI);
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const OMPTraitInfo *TI);
}

void OMPDeclareVariantAttr::printPrettyPragma(
    raw_ostream &OS, const PrintingPolicy &Policy) const {
  if (const Expr *E = getVariantFuncRef()) {
    OS << "(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  }
  OS << " match(" << traitInfos << ")";

  auto PrintExprs = [&OS, &Policy](Expr **Begin, Expr **End) {
    for (Expr **I = Begin; I != End; ++I) {
      assert(*I && "Expected non-null Stmt");
      if (I != Begin)
        OS << ",";
      (*I)->printPretty(OS, nullptr, Policy);
    }
  };
  if (adjustArgsNothing_size()) {
    OS << " adjust_args(nothing:";
    PrintExprs(adjustArgsNothing_begin(), adjustArgsNothing_end());
    OS << ")";
  }
  if (adjustArgsNeedDevicePtr_size()) {
    OS << " adjust_args(need_device_ptr:";
    PrintExprs(adjustArgsNeedDevicePtr_begin(), adjustArgsNeedDevicePtr_end());
    OS << ")";
  }

  auto PrintInteropInfo = [&OS](OMPInteropInfo *Begin, OMPInteropInfo *End) {
    for (OMPInteropInfo *I = Begin; I != End; ++I) {
      if (I != Begin)
        OS << ", ";
      OS << "interop(";
      OS << getInteropTypeString(I);
      OS << ")";
    }
  };
  if (appendArgs_size()) {
    OS << " append_args(";
    PrintInteropInfo(appendArgs_begin(), appendArgs_end());
    OS << ")";
  }
}

#include "clang/AST/AttrImpl.inc"
