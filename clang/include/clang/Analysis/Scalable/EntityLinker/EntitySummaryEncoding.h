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

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_ENTITYSUMMARYENCODING_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_ENTITYSUMMARYENCODING_H

#include "clang/Analysis/Scalable/Model/EntityId.h"
#include <map>

namespace clang::ssaf {

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
  /// \param EntityResolutionTable Mapping from old EntityIds to new EntityIds.
  virtual void
  patch(const std::map<EntityId, EntityId> &EntityResolutionTable) = 0;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_ENTITYSUMMARYENCODING_H
