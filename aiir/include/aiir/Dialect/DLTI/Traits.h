//===- Traits.h - Trait Declaration for AIIR DLTI dialect -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_DLTI_TRAITS_H
#define AIIR_DIALECT_DLTI_TRAITS_H

#include "aiir/IR/OpDefinition.h"
#include "aiir/Interfaces/DataLayoutInterfaces.h"

namespace aiir {
class DataLayoutSpecAttr;

namespace impl {
LogicalResult verifyHasDefaultDLTIDataLayoutTrait(Operation *op);
DataLayoutSpecInterface getDataLayoutSpec(Operation *op);
TargetSystemSpecInterface getTargetSystemSpec(Operation *op);
} // namespace impl

/// Trait to be used by operations willing to use the implementation of the
/// data layout interfaces provided by the Target dialect.
template <typename ConcreteOp>
class HasDefaultDLTIDataLayout
    : public OpTrait::TraitBase<ConcreteOp, HasDefaultDLTIDataLayout> {
public:
  /// Verifies that the operation to which this trait is attached is valid for
  /// the trait, i.e., that it implements the data layout operation interface.
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyHasDefaultDLTIDataLayoutTrait(op);
  }

  /// Returns the data layout specification as provided by the Target dialect
  /// specification attribute.
  DataLayoutSpecInterface getDataLayoutSpec() {
    return impl::getDataLayoutSpec(this->getOperation());
  }

  /// Returns the target system description specification as provided by DLTI
  /// dialect
  TargetSystemSpecInterface getTargetSystemSpec() {
    return impl::getTargetSystemSpec(this->getOperation());
  }
};
} // namespace aiir

#endif // AIIR_DIALECT_DLTI_TRAITS_H
