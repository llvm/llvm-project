//===- ASTEntityMapping.h - AST to SSAF Entity mapping ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ASTENTITYMAPPING_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ASTENTITYMAPPING_H

#include "clang/AST/Decl.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace clang::ssaf {

/// Maps a declaration to an EntityName.
///
/// Supported declaration types for entity mapping:
/// - Functions and methods
/// - Global Variables
/// - Function parameters
/// - Struct/class/union type definitions
/// - Struct/class/union fields
///
/// Implicit declarations and compiler builtins are not mapped.
///
/// \param D The declaration to map. Must not be null.
///
/// \return An EntityName if the declaration can be mapped, std::nullopt
/// otherwise.
std::optional<EntityName> getEntityName(const Decl *D);

/// Maps return entity of a function to an EntityName.
/// The returned name uniquely identifies the return value of function \param
/// FD.
///
/// \param FD The function declaration. Must not be null.
///
/// \return An EntityName for the function's return entity.
std::optional<EntityName> getEntityNameForReturn(const FunctionDecl *FD);

/// Computes the SSAF linkage of a declaration.
///
/// \param D The declaration to classify. Must not be null.
EntityLinkageType getLinkageForDecl(const Decl *D);

/// Returns the EntityName qualified with the build namespaces
/// it would carry after linking into \p LUNamespace.
///
/// \param D The declaration to map. Must not be null.
/// \param TUNamespace The CompilationUnit namespace of a translation unit.
/// \param LUNamespace The LinkUnit namespace the translation unit links into.
/// \return The qualified EntityName if the declaration can be mapped,
/// std::nullopt otherwise.
std::optional<EntityName>
getQualifiedEntityName(const Decl *D, const NestedBuildNamespace &TUNamespace,
                       const NestedBuildNamespace &LUNamespace);

/// Similar to `getQualifiedEntityName`, but for entities of function return
/// values.
std::optional<EntityName>
getQualifiedEntityNameForReturn(const FunctionDecl *FD,
                                const NestedBuildNamespace &TUNamespace,
                                const NestedBuildNamespace &LUNamespace);

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ASTENTITYMAPPING_H
