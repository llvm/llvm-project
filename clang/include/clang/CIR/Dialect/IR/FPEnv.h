//===- FPEnv.h ---- FP Environment ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains the declarations of entities that describe floating
/// point environment and related functions.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_FPENV_H
#define CLANG_CIR_DIALECT_IR_FPENV_H

#include "llvm/ADT/FloatingPointMode.h"

#include <optional>

namespace cir {

namespace fp {

/// Exception behavior used for floating point operations.
///
/// Each of these values corresponds to some LLVMIR metadata argument value of a
/// constrained floating point intrinsic. See the LLVM Language Reference Manual
/// for details.
enum ExceptionBehavior : uint8_t {
  ebIgnore,  ///< This corresponds to "fpexcept.ignore".
  ebMayTrap, ///< This corresponds to "fpexcept.maytrap".
  ebStrict,  ///< This corresponds to "fpexcept.strict".
};

} // namespace fp

/// For any RoundingMode enumerator, returns a string valid as input in
/// constrained intrinsic rounding mode metadata.
std::optional<llvm::StringRef> convertRoundingModeToStr(llvm::RoundingMode);

/// For any ExceptionBehavior enumerator, returns a string valid as input in
/// constrained intrinsic exception behavior metadata.
std::optional<llvm::StringRef>
    convertExceptionBehaviorToStr(fp::ExceptionBehavior);

} // namespace cir

#endif
