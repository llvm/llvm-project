/*===-- UbsanMinimalHandlers.inc - ubsan minimum handlers ------*- C++ -*-=== *\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

// There are two copies of this file, one in LLVM and one in compiler-rt.
// These two MUST BE KEPT IN SYNC.
// * llvm/include/llvm/Transforms/Instrumentation/UbsanMinimalHandlers.inc
// * compiler-rt/lib/ubsan_minimal/UbsanMinimalHandlers.inc

#ifndef UBSAN_HANDLER_PTR
#define UBSAN_HANDLER_PTR(Enum, String)
#endif

#ifndef UBSAN_HANDLER
#define UBSAN_HANDLER(Enum, String)
#endif

#ifndef UBSAN_HANDLER_RECOVER
#define UBSAN_HANDLER_RECOVER(Enum, String)
#endif

UBSAN_HANDLER_PTR(type_mismatch, "type-mismatch")
UBSAN_HANDLER(alignment_assumption, "alignment-assumption")
UBSAN_HANDLER(add_overflow, "add-overflow")
UBSAN_HANDLER(sub_overflow, "sub-overflow")
UBSAN_HANDLER(mul_overflow, "mul-overflow")
UBSAN_HANDLER(negate_overflow, "negate-overflow")
UBSAN_HANDLER(divrem_overflow, "divrem-overflow")
UBSAN_HANDLER(shift_out_of_bounds, "shift-out-of-bounds")
UBSAN_HANDLER(out_of_bounds, "out-of-bounds")
UBSAN_HANDLER(local_out_of_bounds, "local-out-of-bounds")
UBSAN_HANDLER_RECOVER(builtin_unreachable, "builtin-unreachable")
UBSAN_HANDLER_RECOVER(missing_return, "missing-return")
UBSAN_HANDLER(vla_bound_not_positive, "vla-bound-not-positive")
UBSAN_HANDLER(float_cast_overflow, "float-cast-overflow")
UBSAN_HANDLER(load_invalid_value, "load-invalid-value")
UBSAN_HANDLER(invalid_builtin, "invalid-builtin")
UBSAN_HANDLER(invalid_objc_cast, "invalid-objc-cast")
UBSAN_HANDLER(function_type_mismatch, "function-type-mismatch")
UBSAN_HANDLER(implicit_conversion, "implicit-conversion")
UBSAN_HANDLER(nonnull_arg, "nonnull-arg")
UBSAN_HANDLER(nonnull_return, "nonnull-return")
UBSAN_HANDLER(nullability_arg, "nullability-arg")
UBSAN_HANDLER(nullability_return, "nullability-return")
UBSAN_HANDLER(pointer_overflow, "pointer-overflow")
UBSAN_HANDLER(cfi_check_fail, "cfi-check-fail")
