//===- ModelStringConversions.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal string conversion utilities for SSAF model types.
//
// These functions are shared by the model .cpp files (for operator<<) and
// JSONFormat.cpp (for serialization). They are not part of the public API.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_ScalableStaticAnalysis_CORE_MODELSTRINGCONVERSIONS_H
#define LLVM_CLANG_LIB_ScalableStaticAnalysis_CORE_MODELSTRINGCONVERSIONS_H

#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

namespace clang::ssaf {

//===----------------------------------------------------------------------===//
// BuildNamespaceKind
//===----------------------------------------------------------------------===//

/// Returns the canonical string representation of \p BNK used for
/// serialization and display (e.g. "CompilationUnit", "LinkUnit").
inline llvm::StringRef buildNamespaceKindToString(BuildNamespaceKind BNK) {
  switch (BNK) {
  case BuildNamespaceKind::CompilationUnit:
    return "CompilationUnit";
  case BuildNamespaceKind::LinkUnit:
    return "LinkUnit";
  case BuildNamespaceKind::StaticLibrary:
    return "StaticLibrary";
  case BuildNamespaceKind::MultiArchStaticLibrary:
    return "MultiArchStaticLibrary";
  }
  llvm_unreachable("Unhandled BuildNamespaceKind variant");
}

/// Parses a string produced by buildNamespaceKindToString(). Returns
/// std::nullopt if \p Str does not match any known BuildNamespaceKind value.
inline std::optional<BuildNamespaceKind>
buildNamespaceKindFromString(llvm::StringRef Str) {
  if (Str == "CompilationUnit")
    return BuildNamespaceKind::CompilationUnit;
  if (Str == "LinkUnit")
    return BuildNamespaceKind::LinkUnit;
  if (Str == "StaticLibrary")
    return BuildNamespaceKind::StaticLibrary;
  if (Str == "MultiArchStaticLibrary")
    return BuildNamespaceKind::MultiArchStaticLibrary;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// EntityLinkageType
//===----------------------------------------------------------------------===//

/// Returns the canonical string representation of \p LT used for
/// serialization and display (e.g. "None", "Internal", "External").
inline llvm::StringRef entityLinkageTypeToString(EntityLinkageType LT) {
  switch (LT) {
  case EntityLinkageType::None:
    return "None";
  case EntityLinkageType::Internal:
    return "Internal";
  case EntityLinkageType::External:
    return "External";
  }
  llvm_unreachable("Unhandled EntityLinkageType variant");
}

/// Parses a string produced by entityLinkageTypeToString(). Returns
/// std::nullopt if \p Str does not match any known EntityLinkageType value.
inline std::optional<EntityLinkageType>
entityLinkageTypeFromString(llvm::StringRef Str) {
  if (Str == "None")
    return EntityLinkageType::None;
  if (Str == "Internal")
    return EntityLinkageType::Internal;
  if (Str == "External")
    return EntityLinkageType::External;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// EntityBinding
//===----------------------------------------------------------------------===//

/// Returns the canonical string representation of \p B (e.g. "Strong",
/// "Weak").
inline llvm::StringRef entityBindingToString(EntityBinding B) {
  switch (B) {
  case EntityBinding::Strong:
    return "Strong";
  case EntityBinding::Weak:
    return "Weak";
  case EntityBinding::Common:
    return "Common";
  case EntityBinding::Undefined:
    return "Undefined";
  }
  llvm_unreachable("Unhandled EntityBinding variant");
}

/// Parses a string produced by entityBindingToString(). Returns std::nullopt
/// if \p Str does not match any known EntityBinding value.
inline std::optional<EntityBinding>
entityBindingFromString(llvm::StringRef Str) {
  if (Str == "Strong")
    return EntityBinding::Strong;
  if (Str == "Weak")
    return EntityBinding::Weak;
  if (Str == "Common")
    return EntityBinding::Common;
  if (Str == "Undefined")
    return EntityBinding::Undefined;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// EntityCoalescing
//===----------------------------------------------------------------------===//

/// Returns the canonical string representation of \p C (e.g. "None", "ODR").
inline llvm::StringRef entityCoalescingToString(EntityCoalescing C) {
  switch (C) {
  case EntityCoalescing::None:
    return "None";
  case EntityCoalescing::ODR:
    return "ODR";
  }
  llvm_unreachable("Unhandled EntityCoalescing variant");
}

/// Parses a string produced by entityCoalescingToString(). Returns std::nullopt
/// if \p Str does not match any known EntityCoalescing value.
inline std::optional<EntityCoalescing>
entityCoalescingFromString(llvm::StringRef Str) {
  if (Str == "None")
    return EntityCoalescing::None;
  if (Str == "ODR")
    return EntityCoalescing::ODR;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// EntityVisibility
//===----------------------------------------------------------------------===//

/// Returns the canonical string representation of \p V (e.g. "Default",
/// "Hidden", "Protected").
inline llvm::StringRef entityVisibilityToString(EntityVisibility V) {
  switch (V) {
  case EntityVisibility::Default:
    return "Default";
  case EntityVisibility::Hidden:
    return "Hidden";
  case EntityVisibility::Protected:
    return "Protected";
  }
  llvm_unreachable("Unhandled EntityVisibility variant");
}

/// Parses a string produced by entityVisibilityToString(). Returns std::nullopt
/// if \p Str does not match any known EntityVisibility value.
inline std::optional<EntityVisibility>
entityVisibilityFromString(llvm::StringRef Str) {
  if (Str == "Default")
    return EntityVisibility::Default;
  if (Str == "Hidden")
    return EntityVisibility::Hidden;
  if (Str == "Protected")
    return EntityVisibility::Protected;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// EntityDefinitionKind
//===----------------------------------------------------------------------===//

/// Returns the canonical string representation of \p DK (e.g. "Definition",
/// "Declaration").
inline llvm::StringRef entityDefinitionKindToString(EntityDefinitionKind DK) {
  switch (DK) {
  case EntityDefinitionKind::Definition:
    return "Definition";
  case EntityDefinitionKind::Declaration:
    return "Declaration";
  }
  llvm_unreachable("Unhandled EntityDefinitionKind variant");
}

/// Parses a string produced by entityDefinitionKindToString(). Returns
/// std::nullopt if \p Str does not match any known EntityDefinitionKind value.
inline std::optional<EntityDefinitionKind>
entityDefinitionKindFromString(llvm::StringRef Str) {
  if (Str == "Definition")
    return EntityDefinitionKind::Definition;
  if (Str == "Declaration")
    return EntityDefinitionKind::Declaration;
  return std::nullopt;
}

} // namespace clang::ssaf

#endif // LLVM_CLANG_LIB_ScalableStaticAnalysis_CORE_MODELSTRINGCONVERSIONS_H
