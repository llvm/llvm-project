//===-- ubsan_handlers_cxx.cc ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  extern const char *TypeCheckKinds[];
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

  Diag(Loc, DL_Error,
       "%0 address %1 which does not point to an object of type %2")
    << TypeCheckKinds[Data->TypeCheckKind] << (void*)Pointer << Data->Type;

  // If possible, say what type it actually points to.
  if (!DTI.isValid())
    Diag(Pointer, DL_Note, "object has invalid vptr")
        << TypeName(DTI.getMostDerivedTypeName())
        << Range(Pointer, Pointer + sizeof(uptr), "invalid vptr");
  else if (!DTI.getOffset())
    Diag(Pointer, DL_Note, "object is of type %0")
        << TypeName(DTI.getMostDerivedTypeName())
        << Range(Pointer, Pointer + sizeof(uptr), "vptr for %0");
  else
    // FIXME: Find the type at the specified offset, and include that
    //        in the note.
    Diag(Pointer - DTI.getOffset(), DL_Note,
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
void HandleCFIBadType(CFICheckFailData *Data, ValueHandle Vtable,
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
  case CFITCK_ICall:
    Die();
  }

  Diag(Loc, DL_Error, "control flow integrity check for type %0 failed during "
                      "%1 (vtable address %2)")
      << Data->Type << CheckKindStr << (void *)Vtable;

  // If possible, say what type it actually points to.
  if (!DTI.isValid()) {
    const char *module = Symbolizer::GetOrInit()->GetModuleNameForPc(Vtable);
    if (module)
      Diag(Vtable, DL_Note, "invalid vtable in module %0") << module;
    else
      Diag(Vtable, DL_Note, "invalid vtable");
  } else {
    Diag(Vtable, DL_Note, "vtable is of type %0")
        << TypeName(DTI.getMostDerivedTypeName());
  }
}
}  // namespace __ubsan

#endif // CAN_SANITIZE_UB
