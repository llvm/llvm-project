//===- CIROpDefinitions.cpp - Generated CIR op definitions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is the template for the CIROpDefinitions.<N>.cpp shards. Each shard
// defines GET_OP_DEFS_<N> and includes the TableGen-generated definitions of
// one subset of the CIR operations.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "CIROpDefinitions.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace cir;

#include "clang/CIR/Dialect/IR/CIROps.cpp.inc"
