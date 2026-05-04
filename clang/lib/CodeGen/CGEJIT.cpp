//===--- CGEJIT.cpp - EmbeddedJIT CodeGen Metadata ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements EmbeddedJIT metadata generation for LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/AttrKinds.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace clang;
using namespace CodeGen;

/// Emit !ejit.metadata for an ejit_entry or ejit_period_lc function.
void clang::CodeGen::emitEjitFunctionMetadata(CodeGenModule &CGM,
                                              const FunctionDecl *FD,
                                              llvm::Function *F) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  SmallVector<llvm::Metadata *, 8> Entries;

  // ejit_entry
  if (FD->hasAttr<EjitEntryAttr>()) {
    Entries.push_back(llvm::MDNode::get(Ctx,
        llvm::MDString::get(Ctx, "ejit_entry")));
  }

  // ejit_period_lc
  for (const auto *LCA : FD->specific_attrs<EjitPeriodLcAttr>()) {
    Entries.push_back(llvm::MDNode::get(Ctx, {
        llvm::MDString::get(Ctx, "ejit_period_lc"),
        llvm::MDString::get(Ctx, LCA->getPeriodName())
    }));
  }

  // ejit_period_arr_ind (on parameters)
  for (unsigned I = 0; I < FD->getNumParams(); ++I) {
    const ParmVarDecl *PD = FD->getParamDecl(I);
    if (const auto *IdxAttr = PD->getAttr<EjitPeriodArrIndAttr>()) {
      Entries.push_back(llvm::MDNode::get(Ctx, {
          llvm::MDString::get(Ctx, "ejit_period_arr_ind"),
          llvm::MDString::get(Ctx, IdxAttr->getPeriodName()),
          llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), I))
      }));
    }
  }

  if (!Entries.empty()) {
    llvm::MDNode *MD = llvm::MDNode::getDistinct(Ctx, Entries);
    F->setMetadata("ejit.metadata", MD);
  }
}

/// Emit !ejit.metadata for an ejit_period or ejit_period_arr global variable.
void clang::CodeGen::emitEjitGlobalMetadata(CodeGenModule &CGM,
                                            const VarDecl *VD,
                                            llvm::GlobalVariable *GV) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  SmallVector<llvm::Metadata *, 2> Entries;

  // ejit_period
  if (const auto *PA = VD->getAttr<EjitPeriodAttr>()) {
    Entries.push_back(llvm::MDNode::get(Ctx, {
        llvm::MDString::get(Ctx, "ejit_period"),
        llvm::MDString::get(Ctx, PA->getPeriodName())
    }));
  }

  // ejit_period_arr
  if (const auto *PAA = VD->getAttr<EjitPeriodArrAttr>()) {
    uint64_t Size = 0;
    if (const auto *CAT =
            CGM.getContext().getAsConstantArrayType(VD->getType())) {
      Size = CAT->getSize().getZExtValue();
    }
    Entries.push_back(llvm::MDNode::get(Ctx, {
        llvm::MDString::get(Ctx, "ejit_period_arr"),
        llvm::MDString::get(Ctx, PAA->getPeriodName()),
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), Size))
    }));
  }

  if (!Entries.empty()) {
    llvm::MDNode *MD = llvm::MDNode::getDistinct(Ctx, Entries);
    GV->setMetadata("ejit.metadata", MD);
  }
}
