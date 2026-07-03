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

#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

namespace clang::ssaf {

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

#endif // LLVM_CLANG_LIB_ScalableStaticAnalysis_CORE_MODELSTRINGCONVERSIONS_H
