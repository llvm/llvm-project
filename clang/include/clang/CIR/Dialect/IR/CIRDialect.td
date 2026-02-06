//===- CIRDialect.td - CIR dialect -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRDIALECT_TD
#define CLANG_CIR_DIALECT_IR_CIRDIALECT_TD

include "mlir/IR/OpBase.td"

def CIR_Dialect : Dialect {
  let name = "cir";

  // A short one-line summary of our dialect.
  let summary = "A high-level dialect for analyzing and optimizing Clang "
                "supported languages";

  let cppNamespace = "::cir";

  let useDefaultAttributePrinterParser = 1;

  // Enable constant materialization for the CIR dialect. This generates a
  // declaration for the cir::CIRDialect::materializeConstant function. This
  // hook is necessary for canonicalization to properly handle attributes
  // returned by fold methods, allowing them to be materialized as constant
  // operations in the IR.
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    static llvm::StringRef getSourceLanguageAttrName() { return "cir.lang"; }
    static llvm::StringRef getTripleAttrName() { return "cir.triple"; }
    static llvm::StringRef getOptInfoAttrName() { return "cir.opt_info"; }
    static llvm::StringRef getCalleeAttrName() { return "callee"; }
    static llvm::StringRef getNoThrowAttrName() { return "nothrow"; }
    static llvm::StringRef getNoReturnAttrName() { return "noreturn"; }
    static llvm::StringRef getSideEffectAttrName() { return "side_effect"; }
    static llvm::StringRef getReturnsTwiceAttrName() { return "returns_twice"; }
    static llvm::StringRef getColdAttrName() { return "cold"; }
    static llvm::StringRef getHotAttrName() { return "hot"; }
    static llvm::StringRef getNoDuplicatesAttrName() { return "noduplicate"; }
    static llvm::StringRef getConvergentAttrName() { return "convergent"; }
    static llvm::StringRef getNoUnwindAttrName() { return "nounwind"; }
    static llvm::StringRef getModuleLevelAsmAttrName() { return "cir.module_asm"; }
    static llvm::StringRef getGlobalCtorsAttrName() { return "cir.global_ctors"; }
    static llvm::StringRef getGlobalDtorsAttrName() { return "cir.global_dtors"; }
    static llvm::StringRef getOperandSegmentSizesAttrName() { return "operandSegmentSizes"; }
    static llvm::StringRef getNoCallerSavedRegsAttrName() { return "no_caller_saved_registers"; }
    static llvm::StringRef getNoCallbackAttrName() { return "nocallback"; }
    static llvm::StringRef getAllocSizeAttrName() { return "allocsize"; }
    static llvm::StringRef getOptimizeForSizeAttrName() { return "optsize"; }
    static llvm::StringRef getMinSizeAttrName() { return "minsize"; }
    // Note: we have to name this with the underscore instead of the dash like
    // traditional LLVM-IR does, because the LLVM-IR-Dialect doesn't have a way
    // of forming names with a dash instead of underscore in its auto-generated
    // names. TGLexer.cpp(from tablegen) only allows tablegen-names
    // of a [a-zA-Z0-9_] character regex(numbers only if not first), so there is
    // no way to get an underscore into this, even with escaping.
    static llvm::StringRef getModularFormatAttrName() { return "modular_format"; }
    // NoBuiltins means "don't put builtins into my body", whereas "nobuiltin"
    // means "I'm not a builtin, so don't replace me".  This is a subtle
    // difference, but one that reflects Classic Codegen.
    static llvm::StringRef getNoBuiltinsAttrName() { return "nobuiltins"; }
    static llvm::StringRef getNoBuiltinAttrName() { return "nobuiltin"; }
    static llvm::StringRef getTrapFuncNameAttrName() { return "trap_func_name"; }
    static llvm::StringRef getZeroCallUsedRegsAttrName() { return "zero_call_used_regs"; }
    static llvm::StringRef getSaveRegParamsAttrName() { return "save_reg_params"; }
    static llvm::StringRef getDefaultFuncAttrsAttrName() { return "default_func_attrs"; }

    void registerAttributes();
    void registerTypes();

    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type type,
                   mlir::DialectAsmPrinter &printer) const override;
  }];
}

#endif // CLANG_CIR_DIALECT_IR_CIRDIALECT_TD
