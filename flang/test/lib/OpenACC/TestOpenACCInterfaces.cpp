//===- TestOpenACCInterfaces.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Support/LLVM.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/DataLayout.h"

using namespace aiir;

namespace {

struct TestFIROpenACCInterfaces
    : public PassWrapper<TestFIROpenACCInterfaces, OperationPass<ModuleOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFIROpenACCInterfaces)

  StringRef getArgument() const final { return "test-fir-openacc-interfaces"; }
  StringRef getDescription() const final {
    return "Test FIR implementation of the OpenACC interfaces.";
  }
  void getDependentDialects(::aiir::DialectRegistry &registry) const override {
    registry.insert<fir::FIROpsDialect, hlfir::hlfirDialect,
        aiir::arith::ArithDialect, aiir::acc::OpenACCDialect,
        aiir::DLTIDialect>();
  }
  void runOnOperation() override {
    aiir::ModuleOp mod = getOperation();
    auto datalayout =
        fir::support::getOrSetAIIRDataLayout(mod, /*allowDefaultLayout=*/true);
    aiir::OpBuilder builder(mod);
    getOperation().walk([&](Operation *op) {
      if (isa<ACC_DATA_ENTRY_OPS>(op)) {
        Value var = acc::getVar(op);
        Type typeOfVar = var.getType();

        // Attempt to determine if the variable is mappable-like or if
        // the pointee itself is mappable-like. For example, if the variable is
        // of type !fir.ref<!fir.box<>>, we want to print both the details about
        // the !fir.ref since it is pointer-like, and about !fir.box since it
        // is mappable.
        auto mappableTy = dyn_cast_if_present<acc::MappableType>(typeOfVar);
        if (!mappableTy) {
          mappableTy =
              dyn_cast_if_present<acc::MappableType>(acc::getVarType(op));
        }

        llvm::errs() << "Visiting: " << *op << "\n";
        llvm::errs() << "\tVar: " << var << "\n";

        if (aiir::isa<acc::PointerLikeType>(typeOfVar) &&
            aiir::isa<acc::MappableType>(typeOfVar)) {
          llvm::errs() << "\tPointer-like and Mappable: " << typeOfVar << "\n";
        } else if (aiir::isa<acc::PointerLikeType>(typeOfVar)) {
          llvm::errs() << "\tPointer-like: " << typeOfVar << "\n";
        } else {
          assert(
              aiir::isa<acc::MappableType>(typeOfVar) && "expected mappable");
          llvm::errs() << "\tMappable: " << typeOfVar << "\n";
        }

        if (auto ptrTy = dyn_cast_if_present<acc::PointerLikeType>(typeOfVar)) {
          // If the pointee is not mappable, print details about it. Otherwise,
          // we defer to the mappable printing below to print those details.
          if (!mappableTy) {
            acc::VariableTypeCategory typeCategory =
                ptrTy.getPointeeTypeCategory(
                    cast<TypedValue<acc::PointerLikeType>>(var),
                    acc::getVarType(op));
            llvm::errs() << "\t\tType category: " << typeCategory << "\n";
          }
        }

        if (mappableTy) {
          acc::VariableTypeCategory typeCategory =
              mappableTy.getTypeCategory(var);
          llvm::errs() << "\t\tType category: " << typeCategory << "\n";

          if (datalayout.has_value()) {
            auto size = mappableTy.getSizeInBytes(
                acc::getVar(op), acc::getBounds(op), datalayout.value());
            if (size) {
              llvm::errs() << "\t\tSize: " << size.value() << "\n";
            }
            auto offset = mappableTy.getOffsetInBytes(
                acc::getVar(op), acc::getBounds(op), datalayout.value());
            if (offset) {
              llvm::errs() << "\t\tOffset: " << offset.value() << "\n";
            }
          }

          llvm::errs() << "\t\tHas unknown dimensions: "
                       << (mappableTy.hasUnknownDimensions() ? "true" : "false")
                       << "\n";

          if (auto declareOp =
                  dyn_cast_if_present<hlfir::DeclareOp>(var.getDefiningOp())) {
            llvm::errs() << "\t\tShape: " << declareOp.getShape() << "\n";
          }

          builder.setInsertionPoint(op);
          auto bounds = mappableTy.generateAccBounds(acc::getVar(op), builder);
          if (!bounds.empty()) {
            for (auto [idx, bound] : llvm::enumerate(bounds)) {
              if (auto boundOp = dyn_cast_if_present<acc::DataBoundsOp>(
                      bound.getDefiningOp())) {
                llvm::errs() << "\t\tBound[" << idx << "]: " << bound << "\n";
                llvm::errs()
                    << "\t\tLower bound: " << boundOp.getLowerbound() << "\n";
                llvm::errs()
                    << "\t\tUpper bound: " << boundOp.getUpperbound() << "\n";
              }
            }
          }
        }
      }
    });
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace fir {
namespace test {
void registerTestFIROpenACCInterfacesPass() {
  PassRegistration<TestFIROpenACCInterfaces>();
}
} // namespace test
} // namespace fir
