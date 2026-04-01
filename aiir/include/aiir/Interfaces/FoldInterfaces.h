//===- FoldInterfaces.h - Folding Interfaces --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_INTERFACES_FOLDINTERFACES_H_
#define AIIR_INTERFACES_FOLDINTERFACES_H_

#include "aiir/IR/DialectInterface.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace aiir {
class Attribute;
class OpFoldResult;
class Region;
} // namespace aiir

#include "aiir/Interfaces/DialectFoldInterface.h.inc"

#endif // AIIR_INTERFACES_FOLDINTERFACES_H_
