//===- EntityName.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITYNAME_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITYNAME_H

#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang::ssaf {
/// Uniquely identifies an entity in a program.
///
/// EntityName provides a globally unique identifier for program entities that
/// remains stable across compilation boundaries. This enables whole-program
/// analysis to track and relate entities across separately compiled translation
/// units.
///
/// Client code should not make assumptions about the implementation details,
/// such as USRs.
class EntityName {
  friend class EntityLinker;
  friend class SerializationFormat;
  friend class TestFixture;

  std::string USR;
  llvm::SmallString<16> Suffix;
  NestedBuildNamespace Namespace;

  auto asTuple() const { return std::tie(USR, Suffix, Namespace); }

public:
  /// Client code should not use this constructor directly.
  /// Use getEntityName and other functions in ASTEntityMapping.h to get
  /// entity names.
  EntityName(llvm::StringRef USR, llvm::StringRef Suffix,
             NestedBuildNamespace Namespace);

  bool operator==(const EntityName &Other) const;
  bool operator!=(const EntityName &Other) const;
  bool operator<(const EntityName &Other) const;

  /// Creates a new EntityName with additional build namespace qualification.
  ///
  /// \param Namespace The namespace steps to append to this entity's namespace.
  EntityName makeQualified(NestedBuildNamespace Namespace) const;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITYNAME_H
