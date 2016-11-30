//===-- asan_debugging.cc -------------------------------------------------===//
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
// This file contains various functions that are generally useful to call when
// using a debugger (LLDB, GDB).
//===----------------------------------------------------------------------===//

#include "asan_allocator.h"
#include "asan_descriptions.h"
#include "asan_flags.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_thread.h"

namespace {
using namespace __asan;

void FindInfoForStackVar(uptr addr, const char *frame_descr, uptr offset,
                         AddressDescription *descr) {
  InternalMmapVector<StackVarDescr> vars(16);
  if (!ParseFrameDescription(frame_descr, &vars)) {
    return;
  }

  for (uptr i = 0; i < vars.size(); i++) {
    if (offset <= vars[i].beg + vars[i].size) {
      // We use name_len + 1 because strlcpy will guarantee a \0 at the end, so
      // if we're limiting the copy due to name_len, we add 1 to ensure we copy
      // the whole name and then terminate with '\0'.
      internal_strlcpy(descr->name, vars[i].name_pos,
                       Min(descr->name_size, vars[i].name_len + 1));
      descr->region_address = addr - (offset - vars[i].beg);
      descr->region_size = vars[i].size;
      return;
    }
  }
}

void AsanLocateAddress(uptr addr, AddressDescription *descr) {
  ShadowAddressDescription shadow_descr;
  if (GetShadowAddressInformation(addr, &shadow_descr)) {
    descr->region_kind = ShadowNames[shadow_descr.kind];
    return;
  }
  GlobalAddressDescription global_descr;
  if (GetGlobalAddressInformation(addr, &global_descr)) {
    descr->region_kind = "global";
    auto &g = global_descr.globals[0];
    internal_strlcpy(descr->name, g.name, descr->name_size);
    descr->region_address = g.beg;
    descr->region_size = g.size;
    return;
  }

  StackAddressDescription stack_descr;
  asanThreadRegistry().Lock();
  if (GetStackAddressInformation(addr, &stack_descr)) {
    asanThreadRegistry().Unlock();
    descr->region_kind = "stack";
    if (!stack_descr.frame_descr) {
      descr->name[0] = 0;
      descr->region_address = 0;
      descr->region_size = 0;
    } else {
      FindInfoForStackVar(addr, stack_descr.frame_descr, stack_descr.offset,
                          descr);
    }
    return;
  }
  asanThreadRegistry().Unlock();

  descr->name[0] = 0;
  HeapAddressDescription heap_descr;
  if (GetHeapAddressInformation(addr, 1, &heap_descr)) {
    descr->region_address = heap_descr.chunk_access.chunk_begin;
    descr->region_size = heap_descr.chunk_access.chunk_size;
    descr->region_kind = "heap";
    return;
  }

  descr->region_address = 0;
  descr->region_size = 0;
  descr->region_kind = "heap-invalid";
}

uptr AsanGetStack(uptr addr, uptr *trace, u32 size, u32 *thread_id,
                         bool alloc_stack) {
  AsanChunkView chunk = FindHeapChunkByAddress(addr);
  if (!chunk.IsValid()) return 0;

  StackTrace stack(nullptr, 0);
  if (alloc_stack) {
    if (chunk.AllocTid() == kInvalidTid) return 0;
    stack = chunk.GetAllocStack();
    if (thread_id) *thread_id = chunk.AllocTid();
  } else {
    if (chunk.FreeTid() == kInvalidTid) return 0;
    stack = chunk.GetFreeStack();
    if (thread_id) *thread_id = chunk.FreeTid();
  }

  if (trace && size) {
    size = Min(size, Min(stack.size, kStackTraceMax));
    for (uptr i = 0; i < size; i++)
      trace[i] = StackTrace::GetPreviousInstructionPc(stack.trace[i]);

    return size;
  }

  return 0;
}

}  // namespace

SANITIZER_INTERFACE_ATTRIBUTE
const char *__asan_locate_address(uptr addr, char *name, uptr name_size,
                                  uptr *region_address, uptr *region_size) {
  AddressDescription descr = { name, name_size, 0, 0, nullptr };
  AsanLocateAddress(addr, &descr);
  if (region_address) *region_address = descr.region_address;
  if (region_size) *region_size = descr.region_size;
  return descr.region_kind;
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __asan_get_alloc_stack(uptr addr, uptr *trace, uptr size, u32 *thread_id) {
  return AsanGetStack(addr, trace, size, thread_id, /* alloc_stack */ true);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __asan_get_free_stack(uptr addr, uptr *trace, uptr size, u32 *thread_id) {
  return AsanGetStack(addr, trace, size, thread_id, /* alloc_stack */ false);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __asan_get_shadow_mapping(uptr *shadow_scale, uptr *shadow_offset) {
  if (shadow_scale)
    *shadow_scale = SHADOW_SCALE;
  if (shadow_offset)
    *shadow_offset = SHADOW_OFFSET;
}
