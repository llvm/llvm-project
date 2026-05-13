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
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/AttrKinds.h"
#include "llvm/ExecutionEngine/EJIT/EJitCommon.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm::ejit;

/// Emit !ejit.metadata for an ejit_entry or ejit_period_lc function.
void clang::CodeGen::emitEjitFunctionMetadata(CodeGenModule &CGM,
                                              const FunctionDecl *FD,
                                              llvm::Function *F) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  SmallVector<llvm::Metadata *, 8> Entries;

  // ejit_entry
  if (FD->hasAttr<EjitEntryAttr>()) {
    Entries.push_back(llvm::MDNode::get(Ctx,
        llvm::MDString::get(Ctx, TAG_EJIT_ENTRY)));
  }

  // ejit_period_lc
  for (const auto *LCA : FD->specific_attrs<EjitPeriodLcAttr>()) {
    Entries.push_back(llvm::MDNode::get(Ctx, {
        llvm::MDString::get(Ctx, TAG_EJIT_PERIOD_LC),
        llvm::MDString::get(Ctx, LCA->getPeriodName())
    }));
  }

  // ejit_period_arr_ind (on parameters)
  for (unsigned I = 0; I < FD->getNumParams(); ++I) {
    const ParmVarDecl *PD = FD->getParamDecl(I);
    if (const auto *IdxAttr = PD->getAttr<EjitPeriodArrIndAttr>()) {
      Entries.push_back(llvm::MDNode::get(Ctx, {
          llvm::MDString::get(Ctx, TAG_EJIT_PERIOD_ARR_IND),
          llvm::MDString::get(Ctx, IdxAttr->getPeriodName()),
          llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), I))
      }));
    }
  }

  if (!Entries.empty()) {
    llvm::MDNode *MD = llvm::MDNode::getDistinct(Ctx, Entries);
    F->setMetadata(MD_EJIT_METADATA, MD);
  }
}

/// Recursively collect byte offsets of ejit_may_const fields in a record type.
static void collectMayConstFieldOffsets(ASTContext &Ctx, const RecordDecl *RD,
                                        uint64_t BaseOffset,
                                        SmallVectorImpl<uint64_t> &Offsets) {
  const ASTRecordLayout &Layout = Ctx.getASTRecordLayout(RD);
  for (const FieldDecl *FD : RD->fields()) {
    if (FD->isBitField())
      continue;
    uint64_t FieldOff = BaseOffset + Layout.getFieldOffset(FD->getFieldIndex()) / 8;
    if (FD->hasAttr<EjitMayConstAttr>())
      Offsets.push_back(FieldOff);
    // Recurse into nested structs
    if (const auto *InnerRD = FD->getType()->getAsRecordDecl())
      collectMayConstFieldOffsets(Ctx, InnerRD, FieldOff, Offsets);
  }
}

/// Emit !ejit.metadata for an ejit_period or ejit_period_arr global variable.
void clang::CodeGen::emitEjitGlobalMetadata(CodeGenModule &CGM,
                                            const VarDecl *VD,
                                            llvm::GlobalVariable *GV) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  SmallVector<llvm::Metadata *, 4> Entries;

  // ejit_period
  if (const auto *PA = VD->getAttr<EjitPeriodAttr>()) {
    Entries.push_back(llvm::MDNode::get(Ctx, {
        llvm::MDString::get(Ctx, TAG_EJIT_PERIOD),
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
        llvm::MDString::get(Ctx, TAG_EJIT_PERIOD_ARR),
        llvm::MDString::get(Ctx, PAA->getPeriodName()),
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), Size))
    }));
  }

  // ejit_may_const_field: encode byte offsets for PASS6 fallback
  QualType VT = VD->getType();
  if (const auto *AT = CGM.getContext().getAsArrayType(VT))
    VT = AT->getElementType();
  if (VT->isPointerType())
    VT = VT->getPointeeType();
  if (const auto *RD = VT->getAsRecordDecl()) {
    if (RD->isCompleteDefinition()) {
      SmallVector<uint64_t, 4> Offsets;
      collectMayConstFieldOffsets(CGM.getContext(), RD, 0, Offsets);
      for (uint64_t Off : Offsets) {
        Entries.push_back(llvm::MDNode::get(Ctx, {
            llvm::MDString::get(Ctx, TAG_EJIT_MAY_CONST_FIELD),
            llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), Off))
        }));
      }
    }
  }

  if (!Entries.empty()) {
    llvm::MDNode *MD = llvm::MDNode::getDistinct(Ctx, Entries);
    GV->setMetadata(MD_EJIT_METADATA, MD);
  }
}
