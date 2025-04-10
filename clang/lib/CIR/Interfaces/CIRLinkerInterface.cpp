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
#include "mlir/Linker/LinkerInterface.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace mlir::link;

//===----------------------------------------------------------------------===//
// CIRSymbolLinkerInterface
//===----------------------------------------------------------------------===//

class CIRSymbolLinkerInterface : public SymbolLinkerInterface {
public:
  using SymbolLinkerInterface::SymbolLinkerInterface;

  bool canBeLinked(Operation *op) const override { llvm_unreachable("NYI"); }

  StringRef getSymbol(Operation *op) const override { llvm_unreachable("NYI"); }

  Conflict findConflict(Operation *src) const override {
    llvm_unreachable("NYI");
  }

  bool isLinkNeeded(Conflict pair, bool forDependency) const override {
    llvm_unreachable("NYI");
  }

  LogicalResult resolveConflict(Conflict pair) override {
    llvm_unreachable("NYI");
  }

  void registerForLink(Operation *op) override { llvm_unreachable("NYI"); }

  LogicalResult initialize(ModuleOp src) override { return success(); }

  LogicalResult link(LinkState &state) const override {
    llvm_unreachable("NYI");
  }

  Operation *materialize(Operation *src, LinkState &state) const override {
    llvm_unreachable("NYI");
 }

  SmallVector<Operation *> dependencies(Operation *op) const override {
    llvm_unreachable("NYI");
  }
};

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void cir::registerLinkerInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, cir::CIRDialect *dialect) {
    dialect->addInterfaces<CIRSymbolLinkerInterface>();
  });
}
