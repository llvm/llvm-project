//===-- tysan.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TypeSanitizer.
//
// TypeSanitizer runtime.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

#include "tysan/tysan.h"

#include <stdint.h>
#include <string.h>

using namespace __sanitizer;
using namespace __tysan;

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void
tysan_set_type_unknown(const void *addr, uptr size) {
  if (tysan_inited)
    internal_memset(shadow_for(addr), 0, size * sizeof(uptr));
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void
tysan_copy_types(const void *daddr, const void *saddr, uptr size) {
  if (tysan_inited)
    internal_memmove(shadow_for(daddr), shadow_for(saddr), size * sizeof(uptr));
}

static const char *getDisplayName(const char *Name) {
  if (Name[0] == '\0')
    return "<anonymous type>";

  // Clang generates tags for C++ types that demangle as typeinfo. Remove the
  // prefix from the generated string.
  const char *TIPrefix = "typeinfo name for ";
  size_t TIPrefixLen = strlen(TIPrefix);

  const char *DName = Symbolizer::GetOrInit()->Demangle(Name);
  if (!internal_strncmp(DName, TIPrefix, TIPrefixLen))
    DName += TIPrefixLen;

  return DName;
}

static void printTDName(tysan_type_descriptor *td) {
  if (((sptr)td) <= 0) {
    Printf("<unknown type>");
    return;
  }

  switch (td->Tag) {
  default:
    CHECK(false && "invalid enum value");
    break;
  case TYSAN_MEMBER_TD:
    printTDName(td->Member.Access);
    if (td->Member.Access != td->Member.Base) {
      Printf(" (in ");
      printTDName(td->Member.Base);
      Printf(" at offset %zu)", td->Member.Offset);
    }
    break;
  case TYSAN_STRUCT_TD:
    Printf("%s", getDisplayName(
                     (char *)(td->Struct.Members + td->Struct.MemberCount)));
    break;
  }
}

static tysan_type_descriptor *getRootTD(tysan_type_descriptor *TD) {
  tysan_type_descriptor *RootTD = TD;

  do {
    RootTD = TD;

    if (TD->Tag == TYSAN_STRUCT_TD) {
      if (TD->Struct.MemberCount > 0)
        TD = TD->Struct.Members[0].Type;
      else
        TD = nullptr;
    } else if (TD->Tag == TYSAN_MEMBER_TD) {
      TD = TD->Member.Access;
    } else {
      CHECK(false && "invalid enum value");
      break;
    }
  } while (TD);

  return RootTD;
}

// Walk up TDA to see if it reaches TDB.
static bool walkAliasTree(tysan_type_descriptor *TDA,
                          tysan_type_descriptor *TDB, uptr OffsetA,
                          uptr OffsetB) {
  do {
    if (TDA == TDB)
      return OffsetA == OffsetB;

    if (TDA->Tag == TYSAN_STRUCT_TD) {
      // Reached root type descriptor.
      if (!TDA->Struct.MemberCount)
        break;

      uptr Idx = 0;
      for (; Idx < TDA->Struct.MemberCount - 1; ++Idx) {
        if (TDA->Struct.Members[Idx].Offset >= OffsetA)
          break;
      }

      // This offset can't be negative. Therefore we must be accessing something
      // before the current type (not legal) or partially inside the last type.
      // In the latter case, we adjust Idx.
      if (TDA->Struct.Members[Idx].Offset > OffsetA) {
        // Trying to access something before the current type.
        if (!Idx)
          return false;

        Idx -= 1;
      }

      OffsetA -= TDA->Struct.Members[Idx].Offset;
      TDA = TDA->Struct.Members[Idx].Type;
    } else {
      CHECK(false && "invalid enum value");
      break;
    }
  } while (TDA);

  return false;
}

// Walk up the tree starting with TDA to see if we reach TDB.
static bool isAliasingLegalUp(tysan_type_descriptor *TDA,
                              tysan_type_descriptor *TDB) {
  uptr OffsetA = 0, OffsetB = 0;
  if (TDB->Tag == TYSAN_MEMBER_TD) {
    OffsetB = TDB->Member.Offset;
    TDB = TDB->Member.Base;
  }

  if (TDA->Tag == TYSAN_MEMBER_TD) {
    OffsetA = TDA->Member.Offset;
    TDA = TDA->Member.Base;
  }

  return walkAliasTree(TDA, TDB, OffsetA, OffsetB);
}

static bool isAliasingLegalWithOffset(tysan_type_descriptor *TDA,
                                      tysan_type_descriptor *TDB,
                                      uptr OffsetB) {
  // This is handled by calls to isAliasingLegalUp.
  if (OffsetB == 0)
    return false;

  // You can't have an offset into a member.
  if (TDB->Tag == TYSAN_MEMBER_TD)
    return false;

  uptr OffsetA = 0;
  if (TDA->Tag == TYSAN_MEMBER_TD) {
    OffsetA = TDA->Member.Offset;
    TDA = TDA->Member.Base;
  }

  // Since the access was partially inside TDB (the shadow), it can be assumed
  // that we are accessing a member in an object. This means that rather than
  // walk up the scalar access TDA to reach an object, we should walk up the
  // object TBD to reach the scalar we are accessing it with. The offsets will
  // still be checked at the end to make sure this alias is legal.
  return walkAliasTree(TDB, TDA, OffsetB, OffsetA);
}

static bool isAliasingLegal(tysan_type_descriptor *TDA,
                            tysan_type_descriptor *TDB, uptr OffsetB = 0) {
  if (TDA == TDB || !TDB || !TDA)
    return true;

  // Aliasing is legal is the two types have different root nodes.
  if (getRootTD(TDA) != getRootTD(TDB))
    return true;

  // TDB may have been adjusted by offset TDAOffset in the caller to point to
  // the outer type. Check for aliasing with and without adjusting for this
  // offset.
  return isAliasingLegalUp(TDA, TDB) || isAliasingLegalUp(TDB, TDA) ||
         isAliasingLegalWithOffset(TDA, TDB, OffsetB);
}

namespace __tysan {
class Decorator : public __sanitizer::SanitizerCommonDecorator {
public:
  Decorator() : SanitizerCommonDecorator() {}
  const char *Warning() { return Red(); }
  const char *Name() { return Green(); }
  const char *End() { return Default(); }
};
} // namespace __tysan

ALWAYS_INLINE
static void reportError(void *Addr, int Size, tysan_type_descriptor *TD,
                        tysan_type_descriptor *OldTD, const char *AccessStr,
                        const char *DescStr, int Offset, uptr pc, uptr bp,
                        uptr sp) {
  Decorator d;
  Printf("%s", d.Warning());
  Report("ERROR: TypeSanitizer: type-aliasing-violation on address %p"
         " (pc %p bp %p sp %p tid %llu)\n",
         Addr, (void *)pc, (void *)bp, (void *)sp, GetTid());
  Printf("%s", d.End());
  Printf("%s of size %d at %p with type ", AccessStr, Size, Addr);

  Printf("%s", d.Name());
  printTDName(TD);
  Printf("%s", d.End());

  Printf(" %s of type ", DescStr);

  Printf("%s", d.Name());
  printTDName(OldTD);
  Printf("%s", d.End());

  if (Offset != 0)
    Printf(" that starts at offset %d\n", Offset);
  else
    Printf("\n");

  if (pc) {
    uptr top = 0;
    uptr bottom = 0;
    if (flags().print_stacktrace)
      GetThreadStackTopAndBottom(false, &top, &bottom);

    bool request_fast = StackTrace::WillUseFastUnwind(true);
    BufferedStackTrace ST;
    ST.Unwind(kStackTraceMax, pc, bp, 0, top, bottom, request_fast);
    ST.Print();
  } else {
    Printf("\n");
  }
}

ALWAYS_INLINE
static void SetShadowType(tysan_type_descriptor *td,
                          tysan_type_descriptor **shadowData,
                          uint64_t AccessSize) {
  *shadowData = td;
  uint64_t shadowDataInt = (uint64_t)shadowData;

  for (uint64_t i = 1; i < AccessSize; ++i) {
    int64_t dataOffset = i << PtrShift();
    int64_t *badShadowData = (int64_t *)(shadowDataInt + dataOffset);
    int64_t badTD = int64_t(i) * -1;
    *badShadowData = badTD;
  }
}

ALWAYS_INLINE
static bool GetNotAllBadTD(uint64_t ShadowDataInt, uint64_t AccessSize) {
  bool notAllBadTD = false;
  for (uint64_t i = 1; i < AccessSize; ++i) {
    int64_t **unkShadowData = (int64_t **)(ShadowDataInt + (i << PtrShift()));
    int64_t *ILdTD = *unkShadowData;
    notAllBadTD = notAllBadTD || (ILdTD != nullptr);
  }
  return notAllBadTD;
}

ALWAYS_INLINE
static bool GetNotAllUnkTD(uint64_t ShadowDataInt, uint64_t AccessSize) {
  bool notAllBadTD = false;
  for (uint64_t i = 1; i < AccessSize; ++i) {
    int64_t *badShadowData = (int64_t *)(ShadowDataInt + (i << PtrShift()));
    int64_t ILdTD = *badShadowData;
    notAllBadTD = notAllBadTD || (ILdTD >= 0);
  }
  return notAllBadTD;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void
__tysan_instrument_mem_inst(char *dest, char *src, uint64_t size,
                            bool needsMemMove) {
  tysan_type_descriptor **destShadowDataPtr = shadow_for(dest);

  if (!src) {
    internal_memset((char *)destShadowDataPtr, 0, size << PtrShift());
    return;
  }

  uint64_t srcInt = (uint64_t)src;
  uint64_t srcShadowInt = ((srcInt & AppMask()) << PtrShift()) + ShadowAddr();
  uint64_t *srcShadow = (uint64_t *)srcShadowInt;

  if (needsMemMove) {
    internal_memmove((char *)destShadowDataPtr, srcShadow, size << PtrShift());
  } else {
    internal_memcpy((char *)destShadowDataPtr, srcShadow, size << PtrShift());
  }
}

ALWAYS_INLINE
static void __tysan_check_internal(void *addr, int size,
                                   tysan_type_descriptor *td, int flags,
                                   uptr pc, uptr bp, uptr sp) {
  bool IsRead = flags & 1;
  bool IsWrite = flags & 2;
  const char *AccessStr;
  if (IsRead && !IsWrite)
    AccessStr = "READ";
  else if (!IsRead && IsWrite)
    AccessStr = "WRITE";
  else
    AccessStr = "ATOMIC UPDATE";

  tysan_type_descriptor **OldTDPtr = shadow_for(addr);
  tysan_type_descriptor *OldTD = *OldTDPtr;
  if (((sptr)OldTD) < 0) {
    int i = -((sptr)OldTD);
    OldTDPtr -= i;
    OldTD = *OldTDPtr;

    if (!isAliasingLegal(td, OldTD, i))
      reportError(addr, size, td, OldTD, AccessStr,
                  "accesses part of an existing object", -i, pc, bp, sp);

    return;
  }

  if (!isAliasingLegal(td, OldTD)) {
    reportError(addr, size, td, OldTD, AccessStr, "accesses an existing object",
                0, pc, bp, sp);
    return;
  }

  // These types are allowed to alias (or the stored type is unknown), report
  // an error if we find an interior type.

  for (int i = 0; i < size; ++i) {
    OldTDPtr = shadow_for((void *)(((uptr)addr) + i));
    OldTD = *OldTDPtr;
    if (((sptr)OldTD) >= 0 && !isAliasingLegal(td, OldTD))
      reportError(addr, size, td, OldTD, AccessStr,
                  "partially accesses an object", i, pc, bp, sp);
  }
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void
__tysan_check(void *addr, int size, tysan_type_descriptor *td, int flags) {
  GET_CALLER_PC_BP_SP;
  __tysan_check_internal(addr, size, td, flags, pc, bp, sp);
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void
__tysan_instrument_with_shadow_update(void *ptr, tysan_type_descriptor *td,
                                      bool sanitizeFunction,
                                      uint64_t accessSize, int flags) {
  tysan_type_descriptor **shadowData = shadow_for(ptr);
  tysan_type_descriptor *loadedTD = *shadowData;
  bool shadowIsNull = loadedTD == nullptr;

  // TODO, sanitizeFunction is known at compile time, so maybe this is split
  // into two different functions
  if (sanitizeFunction) {

    if (td != loadedTD) {

      // We now know that the types did not match (we're on the slow path). If
      // the type is unknown, then set it.
      if (shadowIsNull) {
        // We're about to set the type. Make sure that all bytes in the value
        // are also of unknown type.
        bool isAllUnknownTD = GetNotAllUnkTD((uint64_t)shadowData, accessSize);
        if (isAllUnknownTD) {
          GET_CALLER_PC_BP_SP;
          __tysan_check_internal(ptr, accessSize, td, flags, pc, bp, sp);
        }
        SetShadowType(td, shadowData, accessSize);
      } else {
        GET_CALLER_PC_BP_SP;
        __tysan_check_internal(ptr, accessSize, td, flags, pc, bp, sp);
      }
    } else {
      // We appear to have the right type. Make sure that all other bytes in
      // the type are still marked as interior bytes. If not, call the runtime.
      bool isNotAllBadTD = GetNotAllBadTD((uint64_t)shadowData, accessSize);
      if (isNotAllBadTD) {
        GET_CALLER_PC_BP_SP;
        __tysan_check_internal(ptr, accessSize, td, flags, pc, bp, sp);
      }
    }
  } else if (shadowIsNull) {
    SetShadowType(td, shadowData, accessSize);
  }
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void
__tysan_set_shadow_type(void *ptr, tysan_type_descriptor *td,
                        uint64_t accessSize) {
  // In the mode where writes always set the type, for a write (which does
  // not also read), we just set the type.
  tysan_type_descriptor **shadow = shadow_for(ptr);
  SetShadowType(td, shadow, accessSize);
}

Flags __tysan::flags_data;

SANITIZER_INTERFACE_ATTRIBUTE uptr __tysan_shadow_memory_address;
SANITIZER_INTERFACE_ATTRIBUTE uptr __tysan_app_memory_mask;

#ifdef TYSAN_RUNTIME_VMA
// Runtime detected VMA size.
int __tysan::vmaSize;
#endif

void Flags::SetDefaults() {
#define TYSAN_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "tysan_flags.inc"
#undef TYSAN_FLAG
}

static void RegisterTySanFlags(FlagParser *parser, Flags *f) {
#define TYSAN_FLAG(Type, Name, DefaultValue, Description)                      \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "tysan_flags.inc"
#undef TYSAN_FLAG
}

static void InitializeFlags() {
  SetCommonFlagsDefaults();
  {
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.external_symbolizer_path = GetEnv("TYSAN_SYMBOLIZER_PATH");
    OverrideCommonFlags(cf);
  }

  flags().SetDefaults();

  FlagParser parser;
  RegisterCommonFlags(&parser);
  RegisterTySanFlags(&parser, &flags());
  parser.ParseString(GetEnv("TYSAN_OPTIONS"));
  InitializeCommonFlags();
  if (Verbosity())
    ReportUnrecognizedFlags();
  if (common_flags()->help)
    parser.PrintFlagDescriptions();
}

static void TySanInitializePlatformEarly() {
  AvoidCVE_2016_2143();
#ifdef TYSAN_RUNTIME_VMA
  vmaSize = (MostSignificantSetBitIndex(GET_CURRENT_FRAME()) + 1);
#if defined(__aarch64__) && !SANITIZER_APPLE
  if (vmaSize != 39 && vmaSize != 42 && vmaSize != 48) {
    Printf("FATAL: TypeSanitizer: unsupported VMA range\n");
    Printf("FATAL: Found %d - Supported 39, 42 and 48\n", vmaSize);
    Die();
  }
#endif
#endif

  __sanitizer::InitializePlatformEarly();

  __tysan_shadow_memory_address = ShadowAddr();
  __tysan_app_memory_mask = AppMask();
}

namespace __tysan {
bool tysan_inited = false;
bool tysan_init_is_running;
} // namespace __tysan

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __tysan_init() {
  CHECK(!tysan_init_is_running);
  if (tysan_inited)
    return;
  tysan_init_is_running = true;

  InitializeFlags();
  TySanInitializePlatformEarly();

  InitializeInterceptors();

  if (!MmapFixedNoReserve(ShadowAddr(), AppAddr() - ShadowAddr()))
    Die();

  tysan_init_is_running = false;
  tysan_inited = true;
}

#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"),
               used)) static void (*tysan_init_ptr)() = __tysan_init;
#endif
