//===- ShapeInferenceInterface.h - Interface definitions for ShapeInference -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_
#define AIIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_

#include "aiir/IR/OpDefinition.h"

namespace aiir {
namespace toy {

/// Include the auto-generated declarations.
#include "toy/ShapeInferenceOpInterfaces.h.inc"

} // namespace toy
} // namespace aiir

#endif // AIIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_
