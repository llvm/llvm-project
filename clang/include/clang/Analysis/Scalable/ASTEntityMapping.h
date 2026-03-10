//===- ASTEntityMapping.h - AST to SSAF Entity mapping ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_ASTENTITYMAPPING_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_ASTENTITYMAPPING_H

#include "clang/AST/Decl.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
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

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ASTENTITYMAPPING_H
