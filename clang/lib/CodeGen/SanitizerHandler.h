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

// Trap codes are encoded in trap instructions (e.g. `brk #0x55XX` on AArch64,
// `ud1l 0xXX(%eax), %eax` on x86) that debuggers and crash diagnostic tools
// decode. Keep existing codes stable (never renumber) and append new checks
// with an unused code.
#define LIST_SANITIZER_CHECKS                                                  \
  SANITIZER_CHECK(AddOverflow, add_overflow, 0, 0x00,                          \
                  "Integer addition overflowed")                               \
  SANITIZER_CHECK(BuiltinUnreachable, builtin_unreachable, 0, 0x01,            \
                  "_builtin_unreachable(), execution reached an unreachable "  \
                  "program point")                                             \
  SANITIZER_CHECK(CFICheckFail, cfi_check_fail, 0, 0x02,                       \
                  "Control flow integrity check failed")                       \
  SANITIZER_CHECK(DivremOverflow, divrem_overflow, 0, 0x03,                    \
                  "Integer divide or remainder overflowed")                    \
  SANITIZER_CHECK(DynamicTypeCacheMiss, dynamic_type_cache_miss, 0, 0x04,      \
                  "Dynamic type cache miss, member call made on an object "    \
                  "whose dynamic type differs from the expected type")         \
  SANITIZER_CHECK(FloatCastOverflow, float_cast_overflow, 0, 0x05,             \
                  "Floating-point to integer conversion overflowed")           \
  SANITIZER_CHECK(FunctionTypeMismatch, function_type_mismatch, 0, 0x06,       \
                  "Function called with mismatched signature")                 \
  SANITIZER_CHECK(ImplicitConversion, implicit_conversion, 0, 0x07,            \
                  "Implicit integer conversion overflowed or lost data")       \
  SANITIZER_CHECK(InvalidBuiltin, invalid_builtin, 0, 0x08,                    \
                  "Invalid use of builtin function")                           \
  SANITIZER_CHECK(InvalidObjCCast, invalid_objc_cast, 0, 0x09,                 \
                  "Invalid Objective-C cast")                                  \
  SANITIZER_CHECK(LoadInvalidValue, load_invalid_value, 0, 0x0a,               \
                  "Loaded an invalid or uninitialized value for the type")     \
  SANITIZER_CHECK(MissingReturn, missing_return, 0, 0x0b,                      \
                  "Execution reached the end of a value-returning function "   \
                  "without returning a value")                                 \
  SANITIZER_CHECK(MulOverflow, mul_overflow, 0, 0x0c,                          \
                  "Integer multiplication overflowed")                         \
  SANITIZER_CHECK(NegateOverflow, negate_overflow, 0, 0x0d,                    \
                  "Integer negation overflowed")                               \
  SANITIZER_CHECK(                                                             \
      NullabilityArg, nullability_arg, 0, 0x0e,                                \
      "Passing null as an argument which is annotated with _Nonnull")          \
  SANITIZER_CHECK(NullabilityReturn, nullability_return, 1, 0x0f,              \
                  "Returning null from a function with a return type "         \
                  "annotated with _Nonnull")                                   \
  SANITIZER_CHECK(NonnullArg, nonnull_arg, 0, 0x10,                            \
                  "Passing null pointer as an argument which is declared to "  \
                  "never be null")                                             \
  SANITIZER_CHECK(NonnullReturn, nonnull_return, 1, 0x11,                      \
                  "Returning null pointer from a function which is declared "  \
                  "to never return null")                                      \
  SANITIZER_CHECK(OutOfBounds, out_of_bounds, 0, 0x12,                         \
                  "Array index out of bounds")                                 \
  SANITIZER_CHECK(PointerOverflow, pointer_overflow, 0, 0x13,                  \
                  "Pointer arithmetic overflowed bounds")                      \
  SANITIZER_CHECK(ShiftOutOfBounds, shift_out_of_bounds, 0, 0x14,              \
                  "Shift exponent is too large for the type")                  \
  SANITIZER_CHECK(SubOverflow, sub_overflow, 0, 0x15,                          \
                  "Integer subtraction overflowed")                            \
  SANITIZER_CHECK(TypeMismatch, type_mismatch, 1, 0x16,                        \
                  "Alignment, null, or object-size error")                     \
  SANITIZER_CHECK(AlignmentAssumption, alignment_assumption, 0, 0x17,          \
                  "Alignment assumption violated")                             \
  SANITIZER_CHECK(                                                             \
      VLABoundNotPositive, vla_bound_not_positive, 0, 0x18,                    \
      "Variable length array bound evaluates to non-positive value")           \
  /* BoundsSafety Msg is empty because it is not considered part of UBSan; */  \
  /* therefore, no trap reason is emitted for this case. */                    \
  SANITIZER_CHECK(BoundsSafety, bounds_safety, 0, 0x19, "")                    \
  SANITIZER_CHECK(ReturnAlignmentAssumption, return_alignment_assumption, 0,   \
                  0x1a, "Alignment assumption violated")

enum SanitizerHandler {
#define SANITIZER_CHECK(Enum, Name, Version, Code, Msg) Enum = Code,
  LIST_SANITIZER_CHECKS
#undef SANITIZER_CHECK
};

#endif // LLVM_CLANG_LIB_CODEGEN_SANITIZER_HANDLER_H
