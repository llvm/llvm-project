//===- OpenACCVariableInfo.h - OpenACC Variable Info Attr -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the VariableInfoAttr base class and the IsVariableInfo
// trait.
//
// Any dialect can define Language-specific variable metadata attribute by
// manually attaching the IsVariableInfo trait and using VariableInfoAttr as
// the baseCppClass.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_OPENACC_OPENACCVARIABLEINFO_H
#define AIIR_DIALECT_OPENACC_OPENACCVARIABLEINFO_H

#include "aiir/IR/Attributes.h"

namespace aiir {
namespace acc {
namespace AttributeTrait {
/// Trait attached to attributes that are OpenACC variable info attributes.
template <typename ConcreteType>
struct IsVariableInfo
    : public aiir::AttributeTrait::TraitBase<ConcreteType, IsVariableInfo> {};
} // namespace AttributeTrait

/// Base attribute class for language-specific variable information carried
/// through the OpenACC type interface helpers.
class VariableInfoAttr : public aiir::Attribute {
public:
  using Attribute::Attribute;

  static bool classof(aiir::Attribute attr) {
    return attr.hasTrait<::aiir::acc::AttributeTrait::IsVariableInfo>();
  }
};

} // namespace acc
} // namespace aiir

#endif // AIIR_DIALECT_OPENACC_OPENACCVARIABLEINFO_H
