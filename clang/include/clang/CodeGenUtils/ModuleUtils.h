//===--- ModuleUtils.h - Shared module emission queries ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the AST queries that both classic CodeGen and CIR CodeGen
// need while deciding how a declaration is emitted at module scope, such as
// its linkage and whether it belongs in a COMDAT group.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_MODULEUTILS_H
#define LLVM_CLANG_CODEGENUTILS_MODULEUTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/Basic/Module.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace clang::CodeGenUtils {

/// Determines whether the language options require us to model
/// unwind exceptions.  We treat -fexceptions as mandating this
/// except under the fragile ObjC ABI with only ObjC exceptions
/// enabled.  This means, for example, that C with -fexceptions
/// enables this.
/// Return the AST address space of constant literal, which is used to emit
/// the constant literal as global variable in LLVM IR.
/// Note: This is not necessarily the address space of the constant literal
/// in AST. For address space agnostic language, e.g. C++, constant literal
/// in AST is always in default address space.
LangAS getGlobalConstantAddressSpace(const LangOptions &LangOpts,
                                     const TargetInfo &Target);

bool hasUnwindExceptions(const LangOptions &LangOpts);

/// Check whether \p D is a strong definition, and thus must not be given
/// common linkage.  \p NoCommon reflects -fno-common.
bool isVarDeclStrongDefinition(const ASTContext &Ctx, const VarDecl *D,
                               bool NoCommon);

/// Check whether \p D should be emitted into a COMDAT group.
bool shouldBeInCOMDAT(const ASTContext &Ctx, const Decl &D);

/// The C++20 named modules whose Itanium initializer function the global init
/// function of a translation unit calls before the unit's own initializers,
/// in call order. For a module interface or partition unit \p Primary these
/// are the modules it exports, imports, or imports in its global or private
/// module fragment; for any other unit the modules in \p Imported, its import
/// declarations in order. A header-like module has no initializer function,
/// and a named module that needs none is skipped.
llvm::SmallVector<Module *, 8>
importedModulesToInitialize(Module *Primary, llvm::ArrayRef<Module *> Imported);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_MODULEUTILS_H
