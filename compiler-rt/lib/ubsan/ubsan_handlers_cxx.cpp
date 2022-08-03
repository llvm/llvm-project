//===-- ubsan_handlers_cxx.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Error logging entry points for the UBSan runtime, which are only used for C++
// compilations. This file is permitted to use language features which require
// linking against a C++ ABI library.
//
//===----------------------------------------------------------------------===//

#include "ubsan_platform.h"
#if CAN_SANITIZE_UB
#include "ubsan_handlers.h"
#include "ubsan_handlers_cxx.h"
#include "ubsan_diag.h"
#include "ubsan_type_hash.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_suppressions.h"

using namespace __sanitizer;
using namespace __ubsan;

namespace __ubsan {
  extern const char *const TypeCheckKinds[];
}

// Returns true if UBSan has printed an error report.
static bool HandleDynamicTypeCacheMiss(
    DynamicTypeCacheMissData *Data, ValueHandle Pointer, ValueHandle Hash,
    ReportOptions Opts) {
  if (checkDynamicType((void*)Pointer, Data->TypeInfo, Hash))
    // Just a cache miss. The type matches after all.
    return false;

  // Check if error report should be suppressed.
  DynamicTypeInfo DTI = getDynamicTypeInfoFromObject((void*)Pointer);
  if (DTI.isValid() && IsVptrCheckSuppressed(DTI.getMostDerivedTypeName()))
    return false;

  SourceLocation Loc = Data->Loc.acquire();
  ErrorType ET = ErrorType::DynamicTypeMismatch;
  if (ignoreReport(Loc, Opts, ET))
    return false;

  ScopedReport R(Opts, Loc, ET);

  Diag(Loc, DL_Error, ET,
       "%0 address %1 which does not point to an object of type %2")
    << TypeCheckKinds[Data->TypeCheckKind] << (void*)Pointer << Data->Type;

  // If possible, say what type it actually points to.
  if (!DTI.isValid()) {
    if (DTI.getOffset() < -VptrMaxOffsetToTop || DTI.getOffset() > VptrMaxOffsetToTop) {
      Diag(Pointer, DL_Note, ET,
           "object has a possibly invalid vptr: abs(offset to top) too big")
          << TypeName(DTI.getMostDerivedTypeName())
          << Range(Pointer, Pointer + sizeof(uptr), "possibly invalid vptr");
    } else {
      Diag(Pointer, DL_Note, ET, "object has invalid vptr")
          << TypeName(DTI.getMostDerivedTypeName())
          << Range(Pointer, Pointer + sizeof(uptr), "invalid vptr");
    }
  } else if (!DTI.getOffset())
    Diag(Pointer, DL_Note, ET, "object is of type %0")
        << TypeName(DTI.getMostDerivedTypeName())
        << Range(Pointer, Pointer + sizeof(uptr), "vptr for %0");
  else
    // FIXME: Find the type at the specified offset, and include that
    //        in the note.
    Diag(Pointer - DTI.getOffset(), DL_Note, ET,
         "object is base class subobject at offset %0 within object of type %1")
        << DTI.getOffset() << TypeName(DTI.getMostDerivedTypeName())
        << TypeName(DTI.getSubobjectTypeName())
        << Range(Pointer, Pointer + sizeof(uptr),
                 "vptr for %2 base class of %1");
  return true;
}

void __ubsan::__ubsan_handle_dynamic_type_cache_miss(
    DynamicTypeCacheMissData *Data, ValueHandle Pointer, ValueHandle Hash) {
  GET_REPORT_OPTIONS(false);
  HandleDynamicTypeCacheMiss(Data, Pointer, Hash, Opts);
}
void __ubsan::__ubsan_handle_dynamic_type_cache_miss_abort(
    DynamicTypeCacheMissData *Data, ValueHandle Pointer, ValueHandle Hash) {
  // Note: -fsanitize=vptr is always recoverable.
  GET_REPORT_OPTIONS(false);
  if (HandleDynamicTypeCacheMiss(Data, Pointer, Hash, Opts))
    Die();
}

namespace __ubsan {
void __ubsan_handle_cfi_bad_type(CFICheckFailData *Data, ValueHandle Vtable,
                                 bool ValidVtable, ReportOptions Opts) {
  SourceLocation Loc = Data->Loc.acquire();
  ErrorType ET = ErrorType::CFIBadType;

  if (ignoreReport(Loc, Opts, ET))
    return;

  ScopedReport R(Opts, Loc, ET);
  DynamicTypeInfo DTI = ValidVtable
                            ? getDynamicTypeInfoFromVtable((void *)Vtable)
                            : DynamicTypeInfo(0, 0, 0);

  const char *CheckKindStr;
  switch (Data->CheckKind) {
  case CFITCK_VCall:
    CheckKindStr = "virtual call";
    break;
  case CFITCK_NVCall:
    CheckKindStr = "non-virtual call";
    break;
  case CFITCK_DerivedCast:
    CheckKindStr = "base-to-derived cast";
    break;
  case CFITCK_UnrelatedCast:
    CheckKindStr = "cast to unrelated type";
    break;
  case CFITCK_VMFCall:
    CheckKindStr = "virtual pointer to member function call";
    break;
  case CFITCK_ICall:
  case CFITCK_NVMFCall:
    Die();
  }

  Diag(Loc, DL_Error, ET,
       "control flow integrity check for type %0 failed during "
       "%1 (vtable address %2)")
      << Data->Type << CheckKindStr << (void *)Vtable;

  // If possible, say what type it actually points to.
  if (!DTI.isValid())
    Diag(Vtable, DL_Note, ET, "invalid vtable");
  else
    Diag(Vtable, DL_Note, ET, "vtable is of type %0")
        << TypeName(DTI.getMostDerivedTypeName());

  // If the failure involved different DSOs for the check location and vtable,
  // report the DSO names.
  const char *DstModule = Symbolizer::GetOrInit()->GetModuleNameForPc(Vtable);
  if (!DstModule)
    DstModule = "(unknown)";

  const char *SrcModule = Symbolizer::GetOrInit()->GetModuleNameForPc(Opts.pc);
  if (!SrcModule)
    SrcModule = "(unknown)";

  if (internal_strcmp(SrcModule, DstModule))
    Diag(Loc, DL_Note, ET, "check failed in %0, vtable located in %1")
        << SrcModule << DstModule;
}

static bool handleFunctionTypeMismatch(FunctionTypeMismatchData *Data,
                                       ValueHandle Function,
                                       ValueHandle calleeRTTI,
                                       ValueHandle fnRTTI, ReportOptions Opts) {
  if (checkTypeInfoEquality(reinterpret_cast<void *>(calleeRTTI),
                            reinterpret_cast<void *>(fnRTTI)))
    return false;

  SourceLocation CallLoc = Data->Loc.acquire();
  ErrorType ET = ErrorType::FunctionTypeMismatch;

  if (ignoreReport(CallLoc, Opts, ET))
    return true;

  ScopedReport R(Opts, CallLoc, ET);

  SymbolizedStackHolder FLoc(getSymbolizedLocation(Function));
  const char *FName = FLoc.get()->info.function;
  if (!FName)
    FName = "(unknown)";

  Diag(CallLoc, DL_Error, ET,
       "call to function %0 through pointer to incorrect function type %1")
      << FName << Data->Type;
  Diag(FLoc, DL_Note, ET, "%0 defined here") << FName;
  return true;
}

void __ubsan_handle_function_type_mismatch_v1(FunctionTypeMismatchData *Data,
                                              ValueHandle Function,
                                              ValueHandle calleeRTTI,
                                              ValueHandle fnRTTI) {
  GET_REPORT_OPTIONS(false);
  handleFunctionTypeMismatch(Data, Function, calleeRTTI, fnRTTI, Opts);
}

void __ubsan_handle_function_type_mismatch_v1_abort(
    FunctionTypeMismatchData *Data, ValueHandle Function,
    ValueHandle calleeRTTI, ValueHandle fnRTTI) {
  GET_REPORT_OPTIONS(true);
  if (handleFunctionTypeMismatch(Data, Function, calleeRTTI, fnRTTI, Opts))
    Die();
}
}  // namespace __ubsan

#endif // CAN_SANITIZE_UB
