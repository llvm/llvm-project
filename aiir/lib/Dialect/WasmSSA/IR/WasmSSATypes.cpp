//===- WasmSSAOps.cpp - WasmSSA dialect operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "aiir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/Types.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>

namespace aiir::wasmssa {
#include "aiir/Dialect/WasmSSA/IR/WasmSSATypeConstraints.cpp.inc"
} // namespace aiir::wasmssa
