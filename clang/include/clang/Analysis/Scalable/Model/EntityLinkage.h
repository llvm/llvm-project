//===- EntityLinkage.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITYLINKAGE_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITYLINKAGE_H

namespace clang::ssaf {

/// Represents the linkage properties of an entity in the program model.
///
/// EntityLinkage captures whether an entity has no linkage, internal linkage,
/// or external linkage, which determines its visibility and accessibility
/// across translation units.
class EntityLinkage {
  friend class SerializationFormat;
  friend class TestFixture;

public:
  enum class LinkageType {
    None,     ///< local variables, function parameters
    Internal, ///< static functions/variables, anonymous namespace
    External  ///< globally visible across translation units
  };

  explicit EntityLinkage(LinkageType L) : Linkage(L) {}

  LinkageType getLinkage() const { return Linkage; }

private:
  LinkageType Linkage;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITYLINKAGE_H
