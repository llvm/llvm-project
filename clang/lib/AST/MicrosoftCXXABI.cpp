//===------- MicrosoftCXXABI.cpp - AST support for the Microsoft C++ ABI --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ AST support targeting the Microsoft Visual C++
// ABI.
//
//===----------------------------------------------------------------------===//

#include "CXXABI.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/TargetInfo.h"

using namespace clang;

namespace {
class MicrosoftCXXABI : public CXXABI {
  ASTContext &Context;
public:
  MicrosoftCXXABI(ASTContext &Ctx) : Context(Ctx) { }

  unsigned getMemberPointerSize(const MemberPointerType *MPT) const;

  CallingConv getDefaultMethodCallConv(bool isVariadic) const {
    if (!isVariadic && Context.getTargetInfo().getTriple().getArch() == llvm::Triple::x86)
      return CC_X86ThisCall;
    else
      return CC_C;
  }

  bool isNearlyEmpty(const CXXRecordDecl *RD) const {
    // FIXME: Audit the corners
    if (!RD->isDynamicClass())
      return false;

    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    
    // In the Microsoft ABI, classes can have one or two vtable pointers.
    CharUnits PointerSize = 
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(0));
    return Layout.getNonVirtualSize() == PointerSize ||
      Layout.getNonVirtualSize() == PointerSize * 2;
  }    
};
}

unsigned MicrosoftCXXABI::getMemberPointerSize(const MemberPointerType *MPT) const {
  QualType Pointee = MPT->getPointeeType();
  CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  if (RD->getNumVBases() > 0) {
    if (Pointee->isFunctionType())
      return 3;
    else
      return 2;
  } else if (RD->getNumBases() > 1 && Pointee->isFunctionType())
    return 2;
  return 1;
}

CXXABI *clang::CreateMicrosoftCXXABI(ASTContext &Ctx) {
  return new MicrosoftCXXABI(Ctx);
}

