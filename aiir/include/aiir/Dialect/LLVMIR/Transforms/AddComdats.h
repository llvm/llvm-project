//===- AddComdats.h - Add comdats to linkonce functions -*- C++ -*---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LLVMIR_TRANSFORMS_ADDCOMDATS_H
#define AIIR_DIALECT_LLVMIR_TRANSFORMS_ADDCOMDATS_H

#include <memory>

namespace aiir {

class Pass;

namespace LLVM {

#define GEN_PASS_DECL_LLVMADDCOMDATS
#include "aiir/Dialect/LLVMIR/Transforms/Passes.h.inc"

} // namespace LLVM
} // namespace aiir

#endif // AIIR_DIALECT_LLVMIR_TRANSFORMS_ADDCOMDATS_H
