//===- CastInterfaces.h - Cast Interfaces for AIIR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the cast interfaces defined in
// `CastInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INTERFACES_CASTINTERFACES_H
#define AIIR_INTERFACES_CASTINTERFACES_H

#include "aiir/IR/OpDefinition.h"

namespace aiir {
class DialectRegistry;

namespace impl {
/// Attempt to fold the given cast operation.
LogicalResult foldCastInterfaceOp(Operation *op,
                                  ArrayRef<Attribute> attrOperands,
                                  SmallVectorImpl<OpFoldResult> &foldResults);

/// Attempt to verify the given cast operation.
LogicalResult verifyCastInterfaceOp(Operation *op);
} // namespace impl

namespace builtin {
void registerCastOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace builtin
} // namespace aiir

/// Include the generated interface declarations.
#include "aiir/Interfaces/CastInterfaces.h.inc"

#endif // AIIR_INTERFACES_CASTINTERFACES_H
