//===- EntitySummaryEncoding.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the EntitySummaryEncoding class, which represents
// EntitySummary data in an encoded, format-specific form.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYSUMMARYENCODING_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYSUMMARYENCODING_H

#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "llvm/Support/Error.h"
#include <map>
#include <memory>

namespace clang::ssaf {

class EntitySummaryEncoding;

/// Maps each EntityId in a TU summary to the EntityId it resolved to in the
/// link unit.
using EntityResolutionMap = std::map<EntityId, EntityId>;

/// Maps each entity of one summary to its encoded summary data.
using EntityDataMap =
    std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>;

/// Represents EntitySummary data in its serialized, format-specific encoding.
///
/// This abstract base class allows the entity linker to manipulate serialized
/// entity summary data without knowing the exact schema of the EntitySummary
/// subclass. The primary operation is patching EntityId references when
/// entities are merged during linking.
class EntitySummaryEncoding {
public:
  virtual ~EntitySummaryEncoding() = default;

  /// Updates EntityId references in the encoded data.
  ///
  /// \param Resolution Mapping from old EntityIds to new EntityIds.
  virtual llvm::Error patch(const EntityResolutionMap &Resolution) = 0;

  /// Returns an identifier unique to this encoding's concrete class.
  ///
  /// SSAF is built without RTTI, so \c equals implementations use this to
  /// establish that another encoding has the same type before casting it.
  /// Implementations return the address of a static object scoped to the
  /// implementing class.
  virtual const void *getEncodingKind() const = 0;

  /// Returns true if \p Other encodes equivalent data.
  ///
  /// Only encodings of the same concrete class can compare equal;
  /// implementations return false for any other \c getEncodingKind().
  ///
  /// Used to detect definitions that were required to be identical but are
  /// not. \pre Both operands have been patched into the same EntityId space;
  /// comparing encodings that reference different spaces is meaningless.
  virtual bool equals(const EntitySummaryEncoding &Other) const = 0;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYSUMMARYENCODING_H
