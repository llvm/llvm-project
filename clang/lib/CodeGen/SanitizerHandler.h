//===-- SanitizerHandler.h - Definition of sanitizer handlers ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-function state used for llvm translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_SANITIZER_HANDLER_H
#define LLVM_CLANG_LIB_CODEGEN_SANITIZER_HANDLER_H

#define LIST_SANITIZER_CHECKS                                                  \
  SANITIZER_CHECK(AddOverflow, add_overflow, 0)                                \
  SANITIZER_CHECK(BuiltinUnreachable, builtin_unreachable, 0)                  \
  SANITIZER_CHECK(CFICheckFail, cfi_check_fail, 0)                             \
  SANITIZER_CHECK(DivremOverflow, divrem_overflow, 0)                          \
  SANITIZER_CHECK(DynamicTypeCacheMiss, dynamic_type_cache_miss, 0)            \
  SANITIZER_CHECK(FloatCastOverflow, float_cast_overflow, 0)                   \
  SANITIZER_CHECK(FunctionTypeMismatch, function_type_mismatch, 0)             \
  SANITIZER_CHECK(ImplicitConversion, implicit_conversion, 0)                  \
  SANITIZER_CHECK(InvalidBuiltin, invalid_builtin, 0)                          \
  SANITIZER_CHECK(InvalidObjCCast, invalid_objc_cast, 0)                       \
  SANITIZER_CHECK(LoadInvalidValue, load_invalid_value, 0)                     \
  SANITIZER_CHECK(MissingReturn, missing_return, 0)                            \
  SANITIZER_CHECK(MulOverflow, mul_overflow, 0)                                \
  SANITIZER_CHECK(NegateOverflow, negate_overflow, 0)                          \
  SANITIZER_CHECK(NullabilityArg, nullability_arg, 0)                          \
  SANITIZER_CHECK(NullabilityReturn, nullability_return, 1)                    \
  SANITIZER_CHECK(NonnullArg, nonnull_arg, 0)                                  \
  SANITIZER_CHECK(NonnullReturn, nonnull_return, 1)                            \
  SANITIZER_CHECK(OutOfBounds, out_of_bounds, 0)                               \
  SANITIZER_CHECK(PointerOverflow, pointer_overflow, 0)                        \
  SANITIZER_CHECK(ShiftOutOfBounds, shift_out_of_bounds, 0)                    \
  SANITIZER_CHECK(SubOverflow, sub_overflow, 0)                                \
  SANITIZER_CHECK(TypeMismatch, type_mismatch, 1)                              \
  SANITIZER_CHECK(AlignmentAssumption, alignment_assumption, 0)                \
  SANITIZER_CHECK(VLABoundNotPositive, vla_bound_not_positive, 0)              \
  SANITIZER_CHECK(BoundsSafety, bounds_safety, 0)

enum SanitizerHandler {
#define SANITIZER_CHECK(Enum, Name, Version) Enum,
  LIST_SANITIZER_CHECKS
#undef SANITIZER_CHECK
};

#endif // LLVM_CLANG_LIB_CODEGEN_SANITIZER_HANDLER_H
