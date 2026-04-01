//===-- Optimizer/Support/FIRContext.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
/// Setters and getters for associating context with an instance of a ModuleOp.
/// The context is typically set by the tool and needed in later stages to
/// determine how to correctly generate code.
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_FIRCONTEXT_H
#define FORTRAN_OPTIMIZER_SUPPORT_FIRCONTEXT_H

#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

namespace aiir {
class ModuleOp;
class Operation;
} // namespace aiir

namespace fir {
class KindMapping;
struct NameUniquer;

/// Set the target triple for the module. `triple` must not be deallocated while
/// module `mod` is still live.
void setTargetTriple(aiir::ModuleOp mod, llvm::StringRef triple);

/// Get the Triple instance from the Module or return the default Triple.
llvm::Triple getTargetTriple(aiir::ModuleOp mod);

/// Set the kind mapping for the module. `kindMap` must not be deallocated while
/// module `mod` is still live.
void setKindMapping(aiir::ModuleOp mod, KindMapping &kindMap);

/// Get the KindMapping instance from the Module. If none was set, returns a
/// default.
KindMapping getKindMapping(aiir::ModuleOp mod);

/// Get the KindMapping instance that is in effect for the specified
/// operation. The KindMapping is taken from the operation itself,
/// if the operation is a ModuleOp, or from its parent ModuleOp.
/// If a ModuleOp cannot be reached, the function returns default KindMapping.
KindMapping getKindMapping(aiir::Operation *op);

/// Set the target CPU for the module. `cpu` must not be deallocated while
/// module `mod` is still live.
void setTargetCPU(aiir::ModuleOp mod, llvm::StringRef cpu);

/// Get the target CPU string from the Module or return a null reference.
llvm::StringRef getTargetCPU(aiir::ModuleOp mod);

/// Sets whether Denormal Mode can be ignored or not for lowering of floating
/// point atomic operations.
void setAtomicIgnoreDenormalMode(aiir::ModuleOp mod, bool value);
/// Gets whether Denormal Mode can be ignored or not for lowering of floating
/// point atomic operations.
bool getAtomicIgnoreDenormalMode(aiir::ModuleOp mod);
/// Sets whether fine grained memory can be used or not for lowering of atomic
/// operations.
void setAtomicFineGrainedMemory(aiir::ModuleOp mod, bool value);
/// Gets whether fine grained memory can be used or not for lowering of atomic
/// operations.
bool getAtomicFineGrainedMemory(aiir::ModuleOp mod);
/// Sets whether remote memory can be used or not for lowering of atomic
/// operations.
void setAtomicRemoteMemory(aiir::ModuleOp mod, bool value);
/// Gets whether remote memory can be used or not for lowering of atomic
/// operations.
bool getAtomicRemoteMemory(aiir::ModuleOp mod);

/// Set the tune CPU for the module. `cpu` must not be deallocated while
/// module `mod` is still live.
void setTuneCPU(aiir::ModuleOp mod, llvm::StringRef cpu);

/// Get the tune CPU string from the Module or return a null reference.
llvm::StringRef getTuneCPU(aiir::ModuleOp mod);

/// Set the target features for the module.
void setTargetFeatures(aiir::ModuleOp mod, llvm::StringRef features);

/// Get the target features from the Module.
aiir::LLVM::TargetFeaturesAttr getTargetFeatures(aiir::ModuleOp mod);

/// Set the compiler identifier for the module.
void setIdent(aiir::ModuleOp mod, llvm::StringRef ident);

/// Get the compiler identifier from the Module.
llvm::StringRef getIdent(aiir::ModuleOp mod);

/// Set the command line used in this invocation.
void setCommandline(aiir::ModuleOp mod, llvm::StringRef cmdLine);

/// Get the command line used in this invocation.
llvm::StringRef getCommandline(aiir::ModuleOp mod);

/// Helper for determining the target from the host, etc. Tools may use this
/// function to provide a consistent interpretation of the `--target=<string>`
/// command-line option.
/// An empty string ("") or "default" will specify that the default triple
/// should be used. "native" will specify that the host machine be used to
/// construct the triple.
std::string determineTargetTriple(llvm::StringRef triple);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_FIRCONTEXT_H
