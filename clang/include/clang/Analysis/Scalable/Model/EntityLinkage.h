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

public:
  /// Specifies the type of linkage an entity has.
  enum class LinkageType {
    None,     ///< No linkage (e.g., local variables, function parameters)
    Internal, ///< Internal linkage (static functions/variables, anonymous
              ///< namespace)
    External  ///< External linkage (globally visible across translation units)
  };

  /// Constructs an EntityLinkage with no linkage (default).
  EntityLinkage() : Linkage(LinkageType::None) {}

  /// Constructs an EntityLinkage with the specified linkage type.
  ///
  /// \param L The linkage type to assign to this entity.
  explicit EntityLinkage(LinkageType L) : Linkage(L) {}

  /// Returns the linkage type of this entity.
  ///
  /// \return The LinkageType indicating the entity's linkage.
  LinkageType getLinkage() const { return Linkage; }

private:
  LinkageType Linkage;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITYLINKAGE_H
