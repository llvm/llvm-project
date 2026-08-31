//===-- ubsan_device_report.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Relocate device pointers in the loaded image and replay the existing
// handlers.
//
//===----------------------------------------------------------------------===//

#include "ubsan_device.h"

#include "ubsan_device_hsa.h"
#include "ubsan_diag.h"
#include "ubsan_handlers_internal.h"
#include "ubsan_value.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"

using namespace __sanitizer;

namespace __ubsan {
namespace {

enum : u8 {
  KL_None = 0,
  KL_NoData = 1 << 0,
  KL_ExtraLoc = 1 << 1,
  KL_FloatCast = 1 << 2,
};

struct KindLayout {
  u16 Size;
  u8 LocOff;
  u8 NLoc;
  u8 NType;
  u8 Flags;
};

// SourceLocation is pointer-sized; CFICheckKind is a byte before it.
// offsetof(CFICheckFailData, Loc) is invalid: Type is a reference.
struct CFICheckFailLocPrefix {
  CFITypeCheckKind CheckKind;
  SourceLocation Loc;
};

constexpr u8 kCFILocOff =
    static_cast<u8>(__builtin_offsetof(CFICheckFailLocPrefix, Loc));

#define UBSAN_DEVICE_HANDLER(kind, name, reason, size, locoff, nloc, ntype,    \
                             flags, ...)                                       \
  {size, locoff, nloc, ntype, flags},
constexpr KindLayout kLayout[] = {
#include "ubsan_device_checks.inc"
};

static_assert(ARRAY_SIZE(kLayout) == UBSAN_DEVICE_KIND_COUNT,
              "kLayout must match ubsan_device_checks.inc");
static_assert(sizeof(CFICheckFailLocPrefix) <= sizeof(CFICheckFailData),
              "CFI Loc prefix cannot exceed CFICheckFailData");

// The HSA loader knows the host-side address of any device pointer contained in
// a loaded segment. We can read them directly without copying or VRAM access.
const void *Host(uptr Dev) { return GetHsa().HostAddr(Dev); }

// The associated SourceLocation is a C-string located in the device executable.
bool PatchLoc(void *S, uptr Off) {
  SourceLocation Loc;
  internal_memcpy(&Loc, static_cast<char *>(S) + Off, sizeof(Loc));
  if (Loc.isInvalid())
    return true;
  const void *Filename = Host(reinterpret_cast<uptr>(Loc.getFilename()));
  if (!Filename)
    return false;
  SourceLocation R(static_cast<const char *>(Filename), Loc.getLine(),
                   Loc.getColumn());
  internal_memcpy(static_cast<char *>(S) + Off, &R, sizeof(R));
  return true;
}

// TypeDescriptor lives in the device image.
bool PatchType(void *S, uptr Off) {
  uptr P = 0;
  internal_memcpy(&P, static_cast<char *>(S) + Off, sizeof(P));
  const void *H = Host(P);
  if (!H)
    return false;
  P = reinterpret_cast<uptr>(H);
  internal_memcpy(static_cast<char *>(S) + Off, &P, sizeof(P));
  return true;
}

// Report data contains type and source location information, try to extract it.
bool Relocate(void *S, uptr LocOff, unsigned NLoc, unsigned NType) {
  for (unsigned I = 0; I < NLoc; ++I)
    if (!PatchLoc(S, LocOff + I * sizeof(SourceLocation)))
      return false;
  for (unsigned I = 0; I < NType; ++I) {
    uptr Off = LocOff + NLoc * sizeof(SourceLocation) + I * sizeof(uptr);
    if (!PatchType(S, Off))
      return false;
  }
  return true;
}

// Values wider than a register are passed by pointer into device memory.
bool ByPointer(const TypeDescriptor *T) {
  if (!T)
    return false;
  if (T->isIntegerTy())
    return T->getIntegerBitWidth() > sizeof(ValueHandle) * 8;
  if (T->isFloatTy())
    return T->getFloatBitWidth() > sizeof(ValueHandle) * 8;
  return false;
}

const TypeDescriptor *TypeAt(void *S, uptr LocOff, unsigned NLoc,
                             unsigned TypeIdx) {
  uptr Off = LocOff + NLoc * sizeof(SourceLocation) + TypeIdx * sizeof(uptr);
  uptr P = 0;
  internal_memcpy(&P, static_cast<char *>(S) + Off, sizeof(P));
  return reinterpret_cast<const TypeDescriptor *>(P);
}

// TODO: Copy wide values into the report packet instead of dropping the report.
bool InlineValues(unsigned Kind, void *S, uptr LocOff, unsigned NLoc) {
  auto Ok = [&](unsigned TypeIdx) {
    return !ByPointer(TypeAt(S, LocOff, NLoc, TypeIdx));
  };
  switch (Kind) {
  case UBSAN_DEVICE_add_overflow:
  case UBSAN_DEVICE_sub_overflow:
  case UBSAN_DEVICE_mul_overflow:
  case UBSAN_DEVICE_divrem_overflow:
  case UBSAN_DEVICE_negate_overflow:
  case UBSAN_DEVICE_vla_bound_not_positive:
  case UBSAN_DEVICE_load_invalid_value:
  case UBSAN_DEVICE_float_cast_overflow:
    return Ok(0);
  case UBSAN_DEVICE_shift_out_of_bounds:
  case UBSAN_DEVICE_implicit_conversion:
    return Ok(0) && Ok(1);
  case UBSAN_DEVICE_out_of_bounds:
    return Ok(1);
  default:
    return true;
  }
}

// Try to determine if this is a float V1 format emitted by the instrumentation.
bool LooksLikeFloatV1(void *S, bool *Ok) {
  *Ok = true;
  uptr P = 0;
  internal_memcpy(&P, S, sizeof(P));
  if (!P)
    return false;
  const u8 *B = static_cast<const u8 *>(Host(P));
  if (!B) {
    *Ok = false;
    return false;
  }
  return looksLikeFloatCastOverflowDataV1Bytes(B);
}

// Gathers the data associated with the device report packet to recreate the
// original report so it can be serviced by the host UBSan runtime. Uses the
// pointers in-place so the deduplication through acquire() is re-used.
bool Materialize(const __ubsan_device_report &R, void **Data, ValueHandle *V0,
                 ValueHandle *V1, ValueHandle *V2) {
  *Data = nullptr;
  *V0 = static_cast<ValueHandle>(R.val0);
  *V1 = static_cast<ValueHandle>(R.val1);
  *V2 = static_cast<ValueHandle>(R.val2);
  if (R.kind >= UBSAN_DEVICE_KIND_COUNT)
    return false;

  const KindLayout &L = kLayout[R.kind];
  if (L.Flags & KL_NoData)
    return true;

  void *Live = const_cast<void *>(Host(static_cast<uptr>(R.data)));
  if (!Live)
    return false;
  *Data = Live;

  // Attempt to decode the arguments, failure skips the report.
  // TODO: Need to support by-pointer arguments for f128 / i128 arguments.
  unsigned NLoc = L.NLoc;
  unsigned NType = L.NType;
  if (L.Flags & KL_FloatCast) {
    bool Ok = false;
    NLoc = LooksLikeFloatV1(Live, &Ok) ? 0 : 1;
    if (!Ok)
      return false;
  }
  if (!Relocate(Live, L.LocOff, NLoc, NType))
    return false;
  if (!InlineValues(R.kind, Live, L.LocOff, NLoc))
    return false;

  if (L.Flags & KL_ExtraLoc) {
    void *Extra = const_cast<void *>(Host(static_cast<uptr>(R.val0)));
    if (!Extra || !PatchLoc(Extra, 0))
      return false;
    *V0 = reinterpret_cast<ValueHandle>(Extra);
  }
  return true;
}

// Forward the newly manifested host-side report to the appropriate internal
// handler function.
void Replay(unsigned Kind, void *Data, ValueHandle V0, ValueHandle V1,
            ValueHandle V2, ReportOptions Opts) {
  switch (Kind) {
  case UBSAN_DEVICE_type_mismatch:
    handleTypeMismatchImpl(reinterpret_cast<TypeMismatchData *>(Data), V0,
                           Opts);
    break;
  case UBSAN_DEVICE_alignment_assumption:
    handleAlignmentAssumptionImpl(
        reinterpret_cast<AlignmentAssumptionData *>(Data), V0, V1, V2, Opts);
    break;
  case UBSAN_DEVICE_add_overflow:
    handleIntegerOverflowImpl(reinterpret_cast<OverflowData *>(Data), V0, "+",
                              V1, Opts);
    break;
  case UBSAN_DEVICE_sub_overflow:
    handleIntegerOverflowImpl(reinterpret_cast<OverflowData *>(Data), V0, "-",
                              V1, Opts);
    break;
  case UBSAN_DEVICE_mul_overflow:
    handleIntegerOverflowImpl(reinterpret_cast<OverflowData *>(Data), V0, "*",
                              V1, Opts);
    break;
  case UBSAN_DEVICE_negate_overflow:
    handleNegateOverflowImpl(reinterpret_cast<OverflowData *>(Data), V0, Opts);
    break;
  case UBSAN_DEVICE_divrem_overflow:
    handleDivremOverflowImpl(reinterpret_cast<OverflowData *>(Data), V0, V1,
                             Opts);
    break;
  case UBSAN_DEVICE_shift_out_of_bounds:
    handleShiftOutOfBoundsImpl(reinterpret_cast<ShiftOutOfBoundsData *>(Data),
                               V0, V1, Opts);
    break;
  case UBSAN_DEVICE_out_of_bounds:
    handleOutOfBoundsImpl(reinterpret_cast<OutOfBoundsData *>(Data), V0, Opts);
    break;
  case UBSAN_DEVICE_local_out_of_bounds:
    handleLocalOutOfBoundsImpl(Opts);
    break;
  case UBSAN_DEVICE_vla_bound_not_positive:
    handleVLABoundNotPositive(reinterpret_cast<VLABoundData *>(Data), V0, Opts);
    break;
  case UBSAN_DEVICE_float_cast_overflow:
    handleFloatCastOverflow(Data, V0, Opts);
    break;
  case UBSAN_DEVICE_load_invalid_value:
    handleLoadInvalidValue(reinterpret_cast<InvalidValueData *>(Data), V0,
                           Opts);
    break;
  case UBSAN_DEVICE_implicit_conversion:
    handleImplicitConversion(reinterpret_cast<ImplicitConversionData *>(Data),
                             Opts, V0, V1);
    break;
  case UBSAN_DEVICE_invalid_builtin:
    handleInvalidBuiltin(reinterpret_cast<InvalidBuiltinData *>(Data), Opts);
    break;
  case UBSAN_DEVICE_invalid_objc_cast:
    handleInvalidObjCCast(reinterpret_cast<InvalidObjCCast *>(Data), V0, Opts);
    break;
  case UBSAN_DEVICE_nonnull_arg:
    handleNonNullArg(reinterpret_cast<NonNullArgData *>(Data), Opts, true);
    break;
  case UBSAN_DEVICE_nullability_arg:
    handleNonNullArg(reinterpret_cast<NonNullArgData *>(Data), Opts, false);
    break;
  case UBSAN_DEVICE_nonnull_return:
    handleNonNullReturn(reinterpret_cast<NonNullReturnData *>(Data),
                        reinterpret_cast<SourceLocation *>(V0), Opts, true);
    break;
  case UBSAN_DEVICE_nullability_return:
    handleNonNullReturn(reinterpret_cast<NonNullReturnData *>(Data),
                        reinterpret_cast<SourceLocation *>(V0), Opts, false);
    break;
  case UBSAN_DEVICE_pointer_overflow:
    handlePointerOverflowImpl(reinterpret_cast<PointerOverflowData *>(Data), V0,
                              V1, Opts);
    break;
  case UBSAN_DEVICE_function_type_mismatch:
    handleFunctionTypeMismatch(
        reinterpret_cast<FunctionTypeMismatchData *>(Data), V0, Opts);
    break;
  case UBSAN_DEVICE_cfi_check_fail: {
    auto *D = reinterpret_cast<CFICheckFailData *>(Data);
    // Skip virtual CFI checks that need to walk a non-existent vtable.
    if (D->CheckKind == CFITCK_ICall || D->CheckKind == CFITCK_NVMFCall)
      handleCFIBadIcall(D, V0, Opts);
    break;
  }
  case UBSAN_DEVICE_builtin_unreachable:
    handleBuiltinUnreachableImpl(reinterpret_cast<UnreachableData *>(Data),
                                 Opts);
    break;
  case UBSAN_DEVICE_missing_return:
    handleMissingReturnImpl(reinterpret_cast<UnreachableData *>(Data), Opts);
    break;
  }
}

} // namespace

void PrintDeviceReport(const __ubsan_device_report &R) {
  if (R.kind >= UBSAN_DEVICE_KIND_COUNT)
    return;

  void *Data = nullptr;
  ValueHandle V0 = 0, V1 = 0, V2 = 0;
  {
    Lock L(&UbsanDeviceMutex);
    if (!GetHsa().Ready() || !Materialize(R, &Data, &V0, &V1, &V2)) {
      VReport(1, "%s: could not translate device UBSan data 0x%zx (kind %u)\n",
              SanitizerToolName, (uptr)R.data, (unsigned)R.kind);
      return;
    }
  }

  ReportOptions Opts = {};
  Opts.FromUnrecoverableHandler = R.fatal;
  Opts.pc = static_cast<uptr>(R.pc);
  Opts.bp = 0;
  Opts.FromDevice = true;

  Replay(R.kind, Data, V0, V1, V2, Opts);
  if (R.fatal)
    Die();
}

} // namespace __ubsan
