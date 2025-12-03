//===- EntityName.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_ENTITY_NAME_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_ENTITY_NAME_H

#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang::ssaf {

/// Uniquely identifies an entity in a program.
///
/// EntityName provides a globally unique identifier for program entities that remains
/// stable across compilation boundaries. This enables whole-program analysis to track
/// and relate entities across separately compiled translation units.
class EntityName {
  std::string USR;
  llvm::SmallString<16> Suffix;
  NestedBuildNamespace Namespace;

public:
  EntityName(llvm::StringRef USR, llvm::StringRef Suffix,
             NestedBuildNamespace Namespace);

  bool operator==(const EntityName& Other) const;
  bool operator!=(const EntityName& Other) const;
  bool operator<(const EntityName& Other) const;

  EntityName makeQualified(NestedBuildNamespace Namespace);

  friend class LinkUnitResolution;
  friend class SerializationFormat;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ENTITY_NAME_H
