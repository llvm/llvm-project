//===- TransformAttr.h - Transform Dialect Attribute Definition -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TRANSFORM_IR_TRANSFORMATTRS_H
#define AIIR_DIALECT_TRANSFORM_IR_TRANSFORMATTRS_H

#include "aiir/IR/Attributes.h"
#include "aiir/IR/BuiltinAttributes.h"

#include <cstdint>
#include <optional>

#include "aiir/Dialect/Transform/IR/TransformDialectEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/Transform/IR/TransformAttrs.h.inc"

#endif // AIIR_DIALECT_TRANSFORM_IR_TRANSFORMATTRS_H
