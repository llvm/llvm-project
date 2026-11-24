//===-- ubsan_offload_checks.inc --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Supported UBSan check kinds on the target device. Defines the channel ABI.
//
//===----------------------------------------------------------------------===//

#ifndef UBSAN_OFFLOAD_HANDLER
#error "ubsan_offload_checks.inc requires UBSAN_OFFLOAD_HANDLER"
#endif
#ifndef UBSAN_OFFLOAD_HANDLER_NORETURN
#define UBSAN_OFFLOAD_HANDLER_NORETURN UBSAN_OFFLOAD_HANDLER
#endif

UBSAN_OFFLOAD_HANDLER(type_mismatch, type_mismatch_v1, "type mismatch",
                      sizeof(TypeMismatchData), 0, 1, 1, KL_None,
                      (TypeMismatchData * D, ValueHandle P), D, P)
UBSAN_OFFLOAD_HANDLER(alignment_assumption, alignment_assumption,
                      "alignment assumption", sizeof(AlignmentAssumptionData),
                      0, 2, 1, KL_None,
                      (AlignmentAssumptionData * D, ValueHandle P,
                       ValueHandle A, ValueHandle O),
                      D, P, A, O)
UBSAN_OFFLOAD_HANDLER(add_overflow, add_overflow, "integer overflow",
                      sizeof(OverflowData), 0, 1, 1, KL_None,
                      (OverflowData * D, ValueHandle L, ValueHandle R), D, L, R)
UBSAN_OFFLOAD_HANDLER(sub_overflow, sub_overflow, "integer overflow",
                      sizeof(OverflowData), 0, 1, 1, KL_None,
                      (OverflowData * D, ValueHandle L, ValueHandle R), D, L, R)
UBSAN_OFFLOAD_HANDLER(mul_overflow, mul_overflow, "integer overflow",
                      sizeof(OverflowData), 0, 1, 1, KL_None,
                      (OverflowData * D, ValueHandle L, ValueHandle R), D, L, R)
UBSAN_OFFLOAD_HANDLER(negate_overflow, negate_overflow, "negation overflow",
                      sizeof(OverflowData), 0, 1, 1, KL_None,
                      (OverflowData * D, ValueHandle V), D, V)
UBSAN_OFFLOAD_HANDLER(divrem_overflow, divrem_overflow, "division overflow",
                      sizeof(OverflowData), 0, 1, 1, KL_None,
                      (OverflowData * D, ValueHandle L, ValueHandle R), D, L, R)
UBSAN_OFFLOAD_HANDLER(shift_out_of_bounds, shift_out_of_bounds,
                      "shift out of bounds", sizeof(ShiftOutOfBoundsData), 0, 1,
                      2, KL_None,
                      (ShiftOutOfBoundsData * D, ValueHandle L, ValueHandle R),
                      D, L, R)
UBSAN_OFFLOAD_HANDLER(out_of_bounds, out_of_bounds, "array index out of bounds",
                      sizeof(OutOfBoundsData), 0, 1, 2, KL_None,
                      (OutOfBoundsData * D, ValueHandle I), D, I)
UBSAN_OFFLOAD_HANDLER(local_out_of_bounds, local_out_of_bounds,
                      "local array index out of bounds", 0, 0, 0, 0, KL_NoData,
                      (void), nullptr)
UBSAN_OFFLOAD_HANDLER(vla_bound_not_positive, vla_bound_not_positive,
                      "non-positive VLA bound", sizeof(VLABoundData), 0, 1, 1,
                      KL_None, (VLABoundData * D, ValueHandle B), D, B)
UBSAN_OFFLOAD_HANDLER(float_cast_overflow, float_cast_overflow,
                      "float cast overflow", sizeof(FloatCastOverflowDataV2), 0,
                      1, 2, KL_FloatCast, (void *D, ValueHandle F), D, F)
UBSAN_OFFLOAD_HANDLER(load_invalid_value, load_invalid_value,
                      "invalid value load", sizeof(InvalidValueData), 0, 1, 1,
                      KL_None, (InvalidValueData * D, ValueHandle V), D, V)
UBSAN_OFFLOAD_HANDLER(
    implicit_conversion, implicit_conversion, "implicit conversion",
    sizeof(ImplicitConversionData), 0, 1, 2, KL_None,
    (ImplicitConversionData * D, ValueHandle S, ValueHandle T), D, S, T)
UBSAN_OFFLOAD_HANDLER(invalid_builtin, invalid_builtin, "invalid builtin",
                      sizeof(InvalidBuiltinData), 0, 1, 0, KL_None,
                      (InvalidBuiltinData * D), D)
UBSAN_OFFLOAD_HANDLER(invalid_objc_cast, invalid_objc_cast, "invalid ObjC cast",
                      sizeof(InvalidObjCCast), 0, 1, 1, KL_None,
                      (InvalidObjCCast * D, ValueHandle P), D, P)
UBSAN_OFFLOAD_HANDLER(nonnull_arg, nonnull_arg, "null argument",
                      sizeof(NonNullArgData), 0, 2, 0, KL_None,
                      (NonNullArgData * D), D)
UBSAN_OFFLOAD_HANDLER(nullability_arg, nullability_arg, "null argument",
                      sizeof(NonNullArgData), 0, 2, 0, KL_None,
                      (NonNullArgData * D), D)
UBSAN_OFFLOAD_HANDLER(nonnull_return, nonnull_return_v1, "null return",
                      sizeof(NonNullReturnData), 0, 1, 0, KL_ExtraLoc,
                      (NonNullReturnData * D, SourceLocation *L), D,
                      reinterpret_cast<ValueHandle>(L))
UBSAN_OFFLOAD_HANDLER(nullability_return, nullability_return_v1, "null return",
                      sizeof(NonNullReturnData), 0, 1, 0, KL_ExtraLoc,
                      (NonNullReturnData * D, SourceLocation *L), D,
                      reinterpret_cast<ValueHandle>(L))
UBSAN_OFFLOAD_HANDLER(pointer_overflow, pointer_overflow, "pointer overflow",
                      sizeof(PointerOverflowData), 0, 1, 0, KL_None,
                      (PointerOverflowData * D, ValueHandle B, ValueHandle R),
                      D, B, R)
UBSAN_OFFLOAD_HANDLER(function_type_mismatch, function_type_mismatch,
                      "function type mismatch",
                      sizeof(FunctionTypeMismatchData), 0, 1, 1, KL_None,
                      (FunctionTypeMismatchData * D, ValueHandle V), D, V)
UBSAN_OFFLOAD_HANDLER(cfi_check_fail, cfi_check_fail,
                      "control flow integrity failure",
                      sizeof(CFICheckFailData), kCFILocOff, 1, 1, KL_None,
                      (CFICheckFailData * D, ValueHandle F, uptr V), D, F, V)
UBSAN_OFFLOAD_HANDLER_NORETURN(builtin_unreachable, builtin_unreachable,
                               "unreachable code reached",
                               sizeof(UnreachableData), 0, 1, 0, KL_None,
                               (UnreachableData * D), D)
UBSAN_OFFLOAD_HANDLER_NORETURN(missing_return, missing_return, "missing return",
                               sizeof(UnreachableData), 0, 1, 0, KL_None,
                               (UnreachableData * D), D)

#undef UBSAN_OFFLOAD_HANDLER
#undef UBSAN_OFFLOAD_HANDLER_NORETURN
