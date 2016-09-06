//===-- asan_descriptions.cc ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan functions for getting information about an address and/or printing it.
//===----------------------------------------------------------------------===//

#include "asan_descriptions.h"
#include "asan_mapping.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

namespace __asan {

// Return " (thread_name) " or an empty string if the name is empty.
const char *ThreadNameWithParenthesis(AsanThreadContext *t, char buff[],
                                      uptr buff_len) {
  const char *name = t->name;
  if (name[0] == '\0') return "";
  buff[0] = 0;
  internal_strncat(buff, " (", 3);
  internal_strncat(buff, name, buff_len - 4);
  internal_strncat(buff, ")", 2);
  return buff;
}

const char *ThreadNameWithParenthesis(u32 tid, char buff[], uptr buff_len) {
  if (tid == kInvalidTid) return "";
  asanThreadRegistry().CheckLocked();
  AsanThreadContext *t = GetThreadContextByTidLocked(tid);
  return ThreadNameWithParenthesis(t, buff, buff_len);
}

void DescribeThread(AsanThreadContext *context) {
  CHECK(context);
  asanThreadRegistry().CheckLocked();
  // No need to announce the main thread.
  if (context->tid == 0 || context->announced) {
    return;
  }
  context->announced = true;
  char tname[128];
  InternalScopedString str(1024);
  str.append("Thread T%d%s", context->tid,
             ThreadNameWithParenthesis(context->tid, tname, sizeof(tname)));
  if (context->parent_tid == kInvalidTid) {
    str.append(" created by unknown thread\n");
    Printf("%s", str.data());
    return;
  }
  str.append(
      " created by T%d%s here:\n", context->parent_tid,
      ThreadNameWithParenthesis(context->parent_tid, tname, sizeof(tname)));
  Printf("%s", str.data());
  StackDepotGet(context->stack_id).Print();
  // Recursively described parent thread if needed.
  if (flags()->print_full_thread_history) {
    AsanThreadContext *parent_context =
        GetThreadContextByTidLocked(context->parent_tid);
    DescribeThread(parent_context);
  }
}

// Shadow descriptions
static bool GetShadowKind(uptr addr, ShadowKind *shadow_kind) {
  CHECK(!AddrIsInMem(addr));
  if (AddrIsInShadowGap(addr)) {
    *shadow_kind = kShadowKindGap;
  } else if (AddrIsInHighShadow(addr)) {
    *shadow_kind = kShadowKindHigh;
  } else if (AddrIsInLowShadow(addr)) {
    *shadow_kind = kShadowKindLow;
  } else {
    CHECK(0 && "Address is not in memory and not in shadow?");
    return false;
  }
  return true;
}

bool DescribeAddressIfShadow(uptr addr) {
  ShadowAddressDescription descr;
  if (!GetShadowAddressInformation(addr, &descr)) return false;
  Printf("Address %p is located in the %s area.\n", addr,
         ShadowNames[descr.kind]);
  return true;
}

bool GetShadowAddressInformation(uptr addr, ShadowAddressDescription *descr) {
  if (AddrIsInMem(addr)) return false;
  ShadowKind shadow_kind;
  if (!GetShadowKind(addr, &shadow_kind)) return false;
  if (shadow_kind != kShadowKindGap) descr->shadow_byte = *(u8 *)addr;
  descr->addr = addr;
  descr->kind = shadow_kind;
  return true;
}

}  // namespace __asan
