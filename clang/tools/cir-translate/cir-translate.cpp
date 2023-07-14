//===- cir-translate.cpp - CIR Translate Driver ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts CIR directly to LLVM IR, similar to mlir-translate or LLVM llc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/Module.h"

namespace cir {
namespace direct {
extern void registerCIRDialectTranslation(mlir::DialectRegistry &registry);
extern std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp theModule,
                             llvm::LLVMContext &llvmCtx);
} // namespace direct
}

void registerToLLVMTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "cir-to-llvmir", "Translate CIR to LLVMIR",
      [](mlir::Operation *op, mlir::raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = cir::direct::lowerDirectlyFromCIRToLLVMIR(
            llvm::dyn_cast<mlir::ModuleOp>(op), llvmContext);
        if (!llvmModule)
          return mlir::failure();
        llvmModule->print(output, nullptr);
        return mlir::success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::DLTIDialect, mlir::func::FuncDialect>();
        mlir::registerAllToLLVMIRTranslations(registry);
        cir::direct::registerCIRDialectTranslation(registry);
      });
}

int main(int argc, char **argv) {
  registerToLLVMTranslation();
  return failed(
      mlir::mlirTranslateMain(argc, argv, "CIR Translation Tool"));
}
