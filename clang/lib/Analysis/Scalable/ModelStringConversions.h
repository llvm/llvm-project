//===- ModelStringConversions.h -------------------------------------------===//
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

#ifndef CLANG_LIB_ANALYSIS_SCALABLE_MODELSTRINGCONVERSIONS_H
#define CLANG_LIB_ANALYSIS_SCALABLE_MODELSTRINGCONVERSIONS_H

#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
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

} // namespace clang::ssaf

#endif // CLANG_LIB_ANALYSIS_SCALABLE_MODELSTRINGCONVERSIONS_H
