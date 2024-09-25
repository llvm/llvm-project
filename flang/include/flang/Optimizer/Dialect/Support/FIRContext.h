//===-- Optimizer/Support/FIRContext.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
/// Setters and getters for associating context with an instance of a ModuleOp.
/// The context is typically set by the tool and needed in later stages to
/// determine how to correctly generate code.
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_FIRCONTEXT_H
#define FORTRAN_OPTIMIZER_SUPPORT_FIRCONTEXT_H

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

namespace mlir {
class ModuleOp;
class Operation;
} // namespace mlir

namespace fir {
class KindMapping;
struct NameUniquer;

/// Set the target triple for the module. `triple` must not be deallocated while
/// module `mod` is still live.
void setTargetTriple(mlir::ModuleOp mod, llvm::StringRef triple);

/// Get the Triple instance from the Module or return the default Triple.
llvm::Triple getTargetTriple(mlir::ModuleOp mod);

/// Set the kind mapping for the module. `kindMap` must not be deallocated while
/// module `mod` is still live.
void setKindMapping(mlir::ModuleOp mod, KindMapping &kindMap);

/// Get the KindMapping instance from the Module. If none was set, returns a
/// default.
KindMapping getKindMapping(mlir::ModuleOp mod);

/// Get the KindMapping instance that is in effect for the specified
/// operation. The KindMapping is taken from the operation itself,
/// if the operation is a ModuleOp, or from its parent ModuleOp.
/// If a ModuleOp cannot be reached, the function returns default KindMapping.
KindMapping getKindMapping(mlir::Operation *op);

/// Set the target CPU for the module. `cpu` must not be deallocated while
/// module `mod` is still live.
void setTargetCPU(mlir::ModuleOp mod, llvm::StringRef cpu);

/// Get the target CPU string from the Module or return a null reference.
llvm::StringRef getTargetCPU(mlir::ModuleOp mod);

/// Set the tune CPU for the module. `cpu` must not be deallocated while
/// module `mod` is still live.
void setTuneCPU(mlir::ModuleOp mod, llvm::StringRef cpu);

/// Get the tune CPU string from the Module or return a null reference.
llvm::StringRef getTuneCPU(mlir::ModuleOp mod);

/// Set the target features for the module.
void setTargetFeatures(mlir::ModuleOp mod, llvm::StringRef features);

/// Get the target features from the Module.
mlir::LLVM::TargetFeaturesAttr getTargetFeatures(mlir::ModuleOp mod);

/// Set the compiler identifier for the module.
void setIdent(mlir::ModuleOp mod, llvm::StringRef ident);

/// Get the compiler identifier from the Module.
llvm::StringRef getIdent(mlir::ModuleOp mod);

/// Helper for determining the target from the host, etc. Tools may use this
/// function to provide a consistent interpretation of the `--target=<string>`
/// command-line option.
/// An empty string ("") or "default" will specify that the default triple
/// should be used. "native" will specify that the host machine be used to
/// construct the triple.
std::string determineTargetTriple(llvm::StringRef triple);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_FIRCONTEXT_H
