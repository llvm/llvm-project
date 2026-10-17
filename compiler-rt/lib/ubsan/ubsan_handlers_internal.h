//===-- ubsan_handlers_internal.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal handle*Impl entry points. Used by the public handlers and by
// device report replay. Not a stable ABI.
//
//===----------------------------------------------------------------------===//

#ifndef UBSAN_HANDLERS_INTERNAL_H
#define UBSAN_HANDLERS_INTERNAL_H

#include "ubsan_diag.h"
#include "ubsan_handlers.h"

namespace __ubsan {

void handleTypeMismatchImpl(TypeMismatchData *Data, ValueHandle Pointer,
                            ReportOptions Opts);
void handleAlignmentAssumptionImpl(AlignmentAssumptionData *Data,
                                   ValueHandle Pointer, ValueHandle Alignment,
                                   ValueHandle Offset, ReportOptions Opts);
void handleIntegerOverflowImpl(OverflowData *Data, ValueHandle LHS,
                               const char *Operator, ValueHandle RHS,
                               ReportOptions Opts);
void handleNegateOverflowImpl(OverflowData *Data, ValueHandle OldVal,
                              ReportOptions Opts);
void handleDivremOverflowImpl(OverflowData *Data, ValueHandle LHS,
                              ValueHandle RHS, ReportOptions Opts);
void handleShiftOutOfBoundsImpl(ShiftOutOfBoundsData *Data, ValueHandle LHS,
                                ValueHandle RHS, ReportOptions Opts);
void handleOutOfBoundsImpl(OutOfBoundsData *Data, ValueHandle Index,
                           ReportOptions Opts);
void handleLocalOutOfBoundsImpl(ReportOptions Opts);
void handleBuiltinUnreachableImpl(UnreachableData *Data, ReportOptions Opts);
void handleMissingReturnImpl(UnreachableData *Data, ReportOptions Opts);
void handleVLABoundNotPositive(VLABoundData *Data, ValueHandle Bound,
                               ReportOptions Opts);
void handleFloatCastOverflow(void *Data, ValueHandle From, ReportOptions Opts);
void handleLoadInvalidValue(InvalidValueData *Data, ValueHandle Val,
                            ReportOptions Opts);
void handleImplicitConversion(ImplicitConversionData *Data, ReportOptions Opts,
                              ValueHandle Src, ValueHandle Dst);
void handleInvalidBuiltin(InvalidBuiltinData *Data, ReportOptions Opts);
void handleInvalidObjCCast(InvalidObjCCast *Data, ValueHandle Pointer,
                           ReportOptions Opts);
void handleNonNullReturn(NonNullReturnData *Data, SourceLocation *LocPtr,
                         ReportOptions Opts, bool IsAttr);
void handleNonNullArg(NonNullArgData *Data, ReportOptions Opts, bool IsAttr);
void handlePointerOverflowImpl(PointerOverflowData *Data, ValueHandle Base,
                               ValueHandle Result, ReportOptions Opts);
void handleCFIBadIcall(CFICheckFailData *Data, ValueHandle Function,
                       ReportOptions Opts);
bool handleFunctionTypeMismatch(FunctionTypeMismatchData *Data,
                                ValueHandle Function, ReportOptions Opts);

} // namespace __ubsan

#endif // UBSAN_HANDLERS_INTERNAL_H
