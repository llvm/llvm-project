//===- EntityLinkage.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITYLINKAGE_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITYLINKAGE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::ssaf {

enum class EntityLinkageType {
  None,     ///< local variables, function parameters
  Internal, ///< static functions/variables, anonymous namespace
  External  ///< globally visible across translation units
};

/// Represents the linkage properties of an entity in the program model.
///
/// EntityLinkage captures whether an entity has no linkage, internal linkage,
/// or external linkage, which determines its visibility and accessibility
/// across translation units.
class EntityLinkage {
  friend class SerializationFormat;
  friend class TestFixture;

public:
  constexpr explicit EntityLinkage(EntityLinkageType L) : Linkage(L) {}

  EntityLinkageType getLinkage() const { return Linkage; }

  bool operator==(const EntityLinkage &Other) const;
  bool operator!=(const EntityLinkage &Other) const;

private:
  EntityLinkageType Linkage;
};

/// Writes a string representation of the linkage type to the stream.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, EntityLinkageType Linkage);

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const EntityLinkage &Linkage);

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_ENTITYLINKAGE_H
