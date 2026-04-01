//===- SPIRVOps.h - AIIR SPIR-V operation traits ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares C++ classes for some of operation traits in the SPIR-V
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SPIRV_IR_SPIRVOPTRAITS_H_
#define AIIR_DIALECT_SPIRV_IR_SPIRVOPTRAITS_H_

#include "aiir/IR/OpDefinition.h"

namespace aiir {
namespace OpTrait {
namespace spirv {

template <typename ConcreteType>
class UnsignedOp : public TraitBase<ConcreteType, UnsignedOp> {};

template <typename ConcreteType>
class SignedOp : public TraitBase<ConcreteType, SignedOp> {};

/// A trait to mark ops that can be enclosed/wrapped in a
/// `SpecConstantOperation` op.
template <typename ConcreteType>
class UsableInSpecConstantOp
    : public TraitBase<ConcreteType, UsableInSpecConstantOp> {};

} // namespace spirv
} // namespace OpTrait
} // namespace aiir

#endif // AIIR_DIALECT_SPIRV_IR_SPIRVOPTRAITS_H_
