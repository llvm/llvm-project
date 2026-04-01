//===- TuneExtensionOps.h - Tune ext. for Transform dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS_H
#define AIIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS_H

#include "aiir/Dialect/Transform/IR/TransformOps.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "aiir/Dialect/Transform/TuneExtension/TuneExtensionOps.h.inc"

#endif // AIIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS_H
