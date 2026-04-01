//===- Import.h - LLVM IR To AIIR translation -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the entry point for the LLVM IR to AIIR conversion.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_IMPORT_H
#define AIIR_TARGET_LLVMIR_IMPORT_H

#include "aiir/IR/OwningOpRef.h"
#include <memory>

// Forward-declare LLVM classes.
namespace llvm {
class DataLayout;
class Module;
} // namespace llvm

namespace aiir {

class DataLayoutSpecInterface;
class AIIRContext;
class ModuleOp;

/// Translates the LLVM module into an AIIR module living in the given context.
/// The translation supports operations from any dialect that has a registered
/// implementation of the LLVMImportDialectInterface. It returns nullptr if the
/// translation fails and reports errors using the error handler registered with
/// the AIIR context.
/// The `emitExpensiveWarnings` option controls if expensive
/// but uncritical diagnostics should be emitted.
/// The `dropDICompositeTypeElements` option controls if DICompositeTypes should
/// be imported without elements. If set, the option avoids the recursive
/// traversal of composite type debug information, which can be expensive for
/// adversarial inputs.
/// The `loadAllDialects` flag (default on) will load all dialects in the
/// context.
/// The `preferUnregisteredIntrinsics` flag (default off) controls whether to
/// import all intrinsics using `llvm.intrinsic_call` even if a dialect
/// registered an explicit intrinsic operation. Warning: passes that rely on
/// matching explicit intrinsic operations may not work properly if this flag is
/// enabled.
/// The `importStructsAsLiterals` flag (default off) ensures that all structs
/// are imported as literal structs, even when they are named in the LLVM
/// module.
OwningOpRef<ModuleOp> translateLLVMIRToModule(
    std::unique_ptr<llvm::Module> llvmModule, AIIRContext *context,
    bool emitExpensiveWarnings = true, bool dropDICompositeTypeElements = false,
    bool loadAllDialects = true, bool preferUnregisteredIntrinsics = false,
    bool importStructsAsLiterals = false);

/// Translate the given LLVM data layout into an AIIR equivalent using the DLTI
/// dialect.
DataLayoutSpecInterface translateDataLayout(const llvm::DataLayout &dataLayout,
                                            AIIRContext *context);

} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_IMPORT_H
