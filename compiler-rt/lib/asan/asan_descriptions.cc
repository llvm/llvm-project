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

// Heap descriptions
static void GetAccessToHeapChunkInformation(ChunkAccess *descr,
                                            AsanChunkView chunk, uptr addr,
                                            uptr access_size) {
  descr->bad_addr = addr;
  if (chunk.AddrIsAtLeft(addr, access_size, &descr->offset)) {
    descr->access_type = kAccessTypeLeft;
  } else if (chunk.AddrIsAtRight(addr, access_size, &descr->offset)) {
    descr->access_type = kAccessTypeRight;
    if (descr->offset < 0) {
      descr->bad_addr -= descr->offset;
      descr->offset = 0;
    }
  } else if (chunk.AddrIsInside(addr, access_size, &descr->offset)) {
    descr->access_type = kAccessTypeInside;
  } else {
    descr->access_type = kAccessTypeUnknown;
  }
  descr->chunk_begin = chunk.Beg();
  descr->chunk_size = chunk.UsedSize();
  descr->alloc_type = chunk.AllocType();
}

static void PrintHeapChunkAccess(uptr addr, const ChunkAccess &descr) {
  Decorator d;
  InternalScopedString str(4096);
  str.append("%s", d.Location());
  switch (descr.access_type) {
    case kAccessTypeLeft:
      str.append("%p is located %zd bytes to the left of",
                 (void *)descr.bad_addr, descr.offset);
      break;
    case kAccessTypeRight:
      str.append("%p is located %zd bytes to the right of",
                 (void *)descr.bad_addr, descr.offset);
      break;
    case kAccessTypeInside:
      str.append("%p is located %zd bytes inside of", (void *)descr.bad_addr,
                 descr.offset);
      break;
    case kAccessTypeUnknown:
      str.append(
          "%p is located somewhere around (this is AddressSanitizer bug!)",
          (void *)descr.bad_addr);
  }
  str.append(" %zu-byte region [%p,%p)\n", descr.chunk_size,
             (void *)descr.chunk_begin,
             (void *)(descr.chunk_begin + descr.chunk_size));
  str.append("%s", d.EndLocation());
  Printf("%s", str.data());
}

bool GetHeapAddressInformation(uptr addr, uptr access_size,
                               HeapAddressDescription *descr) {
  AsanChunkView chunk = FindHeapChunkByAddress(addr);
  if (!chunk.IsValid()) {
    return false;
  }
  descr->addr = addr;
  GetAccessToHeapChunkInformation(&descr->chunk_access, chunk, addr,
                                  access_size);
  CHECK_NE(chunk.AllocTid(), kInvalidTid);
  descr->alloc_tid = chunk.AllocTid();
  descr->alloc_stack_id = chunk.GetAllocStackId();
  descr->free_tid = chunk.FreeTid();
  if (descr->free_tid != kInvalidTid)
    descr->free_stack_id = chunk.GetFreeStackId();
  return true;
}

static StackTrace GetStackTraceFromId(u32 id) {
  CHECK(id);
  StackTrace res = StackDepotGet(id);
  CHECK(res.trace);
  return res;
}

bool DescribeAddressIfHeap(uptr addr, uptr access_size) {
  HeapAddressDescription descr;
  if (!GetHeapAddressInformation(addr, access_size, &descr)) {
    Printf(
        "AddressSanitizer can not describe address in more detail "
        "(wild memory access suspected).\n");
    return false;
  }
  PrintHeapChunkAccess(addr, descr.chunk_access);

  asanThreadRegistry().CheckLocked();
  AsanThreadContext *alloc_thread =
      GetThreadContextByTidLocked(descr.alloc_tid);
  StackTrace alloc_stack = GetStackTraceFromId(descr.alloc_stack_id);

  char tname[128];
  Decorator d;
  AsanThreadContext *free_thread = nullptr;
  if (descr.free_tid != kInvalidTid) {
    free_thread = GetThreadContextByTidLocked(descr.free_tid);
    Printf("%sfreed by thread T%d%s here:%s\n", d.Allocation(),
           free_thread->tid,
           ThreadNameWithParenthesis(free_thread, tname, sizeof(tname)),
           d.EndAllocation());
    StackTrace free_stack = GetStackTraceFromId(descr.free_stack_id);
    free_stack.Print();
    Printf("%spreviously allocated by thread T%d%s here:%s\n", d.Allocation(),
           alloc_thread->tid,
           ThreadNameWithParenthesis(alloc_thread, tname, sizeof(tname)),
           d.EndAllocation());
  } else {
    Printf("%sallocated by thread T%d%s here:%s\n", d.Allocation(),
           alloc_thread->tid,
           ThreadNameWithParenthesis(alloc_thread, tname, sizeof(tname)),
           d.EndAllocation());
  }
  alloc_stack.Print();
  DescribeThread(GetCurrentThread());
  if (free_thread) DescribeThread(free_thread);
  DescribeThread(alloc_thread);
  return true;
}

}  // namespace __asan
