//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to link CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Interfaces/CIRLinkerInterface.h"
#include "mlir/Linker/LLVMLinkerMixin.h"
#include "mlir/Linker/LinkerInterface.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace mlir::link;

//===----------------------------------------------------------------------===//
// CIRSymbolLinkerInterface
//===----------------------------------------------------------------------===//

class CIRSymbolLinkerInterface
    : public SymbolAttrLLVMLinkerInterface<CIRSymbolLinkerInterface> {
public:
  CIRSymbolLinkerInterface(Dialect *dialect)
      : SymbolAttrLLVMLinkerInterface(dialect) {}

  bool canBeLinked(Operation *op) const override {
    return isa<cir::GlobalOp>(op) || isa<cir::FuncOp>(op);
  }

  //===--------------------------------------------------------------------===//
  // LLVMLinkerMixin required methods from derived linker interface
  //===--------------------------------------------------------------------===//

  // TODO: expose convertLinkage from LowerToLLVM.cpp
  static Linkage toLLVMLinkage(cir::GlobalLinkageKind linkage) {
    using CIR = cir::GlobalLinkageKind;
    using LLVM = mlir::LLVM::Linkage;

    switch (linkage) {
    case CIR::AvailableExternallyLinkage:
      return LLVM::AvailableExternally;
    case CIR::CommonLinkage:
      return LLVM::Common;
    case CIR::ExternalLinkage:
      return LLVM::External;
    case CIR::ExternalWeakLinkage:
      return LLVM::ExternWeak;
    case CIR::InternalLinkage:
      return LLVM::Internal;
    case CIR::LinkOnceAnyLinkage:
      return LLVM::Linkonce;
    case CIR::LinkOnceODRLinkage:
      return LLVM::LinkonceODR;
    case CIR::PrivateLinkage:
      return LLVM::Private;
    case CIR::WeakAnyLinkage:
      return LLVM::Weak;
    case CIR::WeakODRLinkage:
      return LLVM::WeakODR;
    };
  }

  static Linkage getLinkage(Operation *op) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return toLLVMLinkage(gv.getLinkage());
    if (auto fn = dyn_cast<cir::FuncOp>(op))
      return toLLVMLinkage(fn.getLinkage());
    llvm_unreachable("unexpected operation");
  }

  // TODO: expose lowerCIRVisibilityToLLVMVisibility from LowerToLLVM.cpp
  static Visibility toLLVMVisibility(cir::VisibilityAttr visibility) {
    return toLLVMVisibility(visibility.getValue());
  }

  static Visibility toLLVMVisibility(cir::VisibilityKind visibility) {
    using CIR = cir::VisibilityKind;
    using LLVM = mlir::LLVM::Visibility;

    switch (visibility) {
    case CIR::Default:
      return LLVM::Default;
    case CIR::Hidden:
      return LLVM::Hidden;
    case CIR::Protected:
      return LLVM::Protected;
    };
  }

  static cir::VisibilityKind toCIRVisibility(Visibility visibility) {
    using CIR = cir::VisibilityKind;
    using LLVM = mlir::LLVM::Visibility;

    switch (visibility) {
    case LLVM::Default:
      return CIR::Default;
    case LLVM::Hidden:
      return CIR::Hidden;
    case LLVM::Protected:
      return CIR::Protected;
    };
  }

  static cir::VisibilityAttr toCIRVisibilityAttr(Visibility visibility,
                                                 MLIRContext *mlirContext) {
    return cir::VisibilityAttr::get(mlirContext, toCIRVisibility(visibility));
  }

  static Visibility getVisibility(Operation *op) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return toLLVMVisibility(gv.getGlobalVisibility());
    if (auto fn = dyn_cast<cir::FuncOp>(op))
      return toLLVMVisibility(fn.getGlobalVisibility());
    llvm_unreachable("unexpected operation");
  }

  static void setVisibility(Operation *op, Visibility visibility) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return gv.setGlobalVisibilityAttr(
          toCIRVisibilityAttr(visibility, op->getContext()));
    if (auto fn = dyn_cast<cir::FuncOp>(op))
      return fn.setGlobalVisibilityAttr(
          toCIRVisibilityAttr(visibility, op->getContext()));
    llvm_unreachable("unexpected operation");
  }

  static bool isDeclaration(Operation *op) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return gv.isDeclaration();
    if (auto fn = dyn_cast<cir::FuncOp>(op))
      return fn.isDeclaration();
    llvm_unreachable("unexpected operation");
  }

  static unsigned getBitWidth(Operation *op) { llvm_unreachable("NYI"); }
};

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void cir::registerLinkerInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, cir::CIRDialect *dialect) {
    dialect->addInterfaces<CIRSymbolLinkerInterface>();
  });
}
