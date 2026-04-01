//===- VectorInterfaces.h - Vector interfaces -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interfaces for vector ops.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INTERFACES_VECTORINTERFACES_H
#define AIIR_INTERFACES_VECTORINTERFACES_H

#include "aiir/IR/AffineMap.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/OpDefinition.h"

/// Include the generated interface declarations.
#include "aiir/Interfaces/VectorInterfaces.h.inc"

#endif // AIIR_INTERFACES_VECTORINTERFACES_H
