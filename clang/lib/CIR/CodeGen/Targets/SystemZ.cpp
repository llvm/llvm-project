//===---- SystemZ.cpp - SystemZ-specific CIR CodeGen ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides SystemZ-specific CIR CodeGen logic.
//
//===----------------------------------------------------------------------===//

#include "ABIInfo.h"
#include "CIRGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/Decl.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

class SystemZABIInfo : public ABIInfo {
public:
  SystemZABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}
};

class SystemZTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  SystemZTargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<SystemZABIInfo>(cgt)) {}

  void setTargetAttributes(const Decl *d, mlir::Operation *gv,
                           CIRGenModule &cgm) const override {
    if (!d)
      return;
    assert(!cir::MissingFeatures::s390xVisibleVectorABI());
  }
};

} // namespace

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createSystemZTargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<SystemZTargetCIRGenInfo>(cgt);
}
