//===- bolt/Core/HashUtilities.h - Misc hash utilities --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions for computing hash values over BinaryFunction
// and BinaryBasicBlock.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_HASH_UTILITIES_H
#define BOLT_CORE_HASH_UTILITIES_H

#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"

namespace llvm {
namespace bolt {

std::string hashInteger(uint64_t Value);

std::string hashSymbol(BinaryContext &BC, const MCSymbol &Symbol);

std::string hashExpr(BinaryContext &BC, const MCExpr &Expr);

std::string hashInstOperand(BinaryContext &BC, const MCOperand &Operand);

using OperandHashFuncTy = function_ref<typename std::string(const MCOperand &)>;

std::string hashBlock(BinaryContext &BC, const BinaryBasicBlock &BB,
                      OperandHashFuncTy OperandHashFunc);

std::string hashBlockLoose(BinaryContext &BC, const BinaryBasicBlock &BB);

} // namespace bolt
} // namespace llvm

#endif
