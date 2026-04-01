//===- IRDLInterfaces.h - IRDL interfaces definition ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interfaces used by the IRDL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_IRDL_IR_IRDLINTERFACES_H_
#define AIIR_DIALECT_IRDL_IR_IRDLINTERFACES_H_

#include "aiir/Dialect/IRDL/IRDLVerifiers.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/ExtensibleDialect.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/Types.h"
#include <optional>

namespace aiir {
namespace irdl {
class TypeOp;
class AttributeOp;
} // namespace irdl
} // namespace aiir

//===----------------------------------------------------------------------===//
// IRDL Dialect Interfaces
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/IRDL/IR/IRDLInterfaces.h.inc"

#endif //  AIIR_DIALECT_IRDL_IR_IRDLINTERFACES_H_
