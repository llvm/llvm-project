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
#include "asan_report.h"
#include "asan_stack.h"
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
  descr.Print();
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
  descr->alloc_type = chunk.GetAllocType();
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
  str.append("%s", d.Default());
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
  descr.Print();
  return true;
}

// Stack descriptions
bool GetStackAddressInformation(uptr addr, uptr access_size,
                                StackAddressDescription *descr) {
  AsanThread *t = FindThreadByStackAddress(addr);
  if (!t) return false;

  descr->addr = addr;
  descr->tid = t->tid();
  // Try to fetch precise stack frame for this access.
  AsanThread::StackFrameAccess access;
  if (!t->GetStackFrameAccessByAddr(addr, &access)) {
    descr->frame_descr = nullptr;
    return true;
  }

  descr->offset = access.offset;
  descr->access_size = access_size;
  descr->frame_pc = access.frame_pc;
  descr->frame_descr = access.frame_descr;

#if SANITIZER_PPC64V1
  // On PowerPC64 ELFv1, the address of a function actually points to a
  // three-doubleword data structure with the first field containing
  // the address of the function's code.
  descr->frame_pc = *reinterpret_cast<uptr *>(descr->frame_pc);
#endif
  descr->frame_pc += 16;

  return true;
}

static void PrintAccessAndVarIntersection(const StackVarDescr &var, uptr addr,
                                          uptr access_size, uptr prev_var_end,
                                          uptr next_var_beg) {
  uptr var_end = var.beg + var.size;
  uptr addr_end = addr + access_size;
  const char *pos_descr = nullptr;
  // If the variable [var.beg, var_end) is the nearest variable to the
  // current memory access, indicate it in the log.
  if (addr >= var.beg) {
    if (addr_end <= var_end)
      pos_descr = "is inside";  // May happen if this is a use-after-return.
    else if (addr < var_end)
      pos_descr = "partially overflows";
    else if (addr_end <= next_var_beg &&
             next_var_beg - addr_end >= addr - var_end)
      pos_descr = "overflows";
  } else {
    if (addr_end > var.beg)
      pos_descr = "partially underflows";
    else if (addr >= prev_var_end && addr - prev_var_end >= var.beg - addr_end)
      pos_descr = "underflows";
  }
  InternalScopedString str(1024);
  str.append("    [%zd, %zd)", var.beg, var_end);
  // Render variable name.
  str.append(" '");
  for (uptr i = 0; i < var.name_len; ++i) {
    str.append("%c", var.name_pos[i]);
  }
  str.append("'");
  if (var.line > 0) {
    str.append(" (line %d)", var.line);
  }
  if (pos_descr) {
    Decorator d;
    // FIXME: we may want to also print the size of the access here,
    // but in case of accesses generated by memset it may be confusing.
    str.append("%s <== Memory access at offset %zd %s this variable%s\n",
               d.Location(), addr, pos_descr, d.Default());
  } else {
    str.append("\n");
  }
  Printf("%s", str.data());
}

bool DescribeAddressIfStack(uptr addr, uptr access_size) {
  StackAddressDescription descr;
  if (!GetStackAddressInformation(addr, access_size, &descr)) return false;
  descr.Print();
  return true;
}

// Global descriptions
static void DescribeAddressRelativeToGlobal(uptr addr, uptr access_size,
                                            const __asan_global &g) {
  InternalScopedString str(4096);
  Decorator d;
  str.append("%s", d.Location());
  if (addr < g.beg) {
    str.append("%p is located %zd bytes to the left", (void *)addr,
               g.beg - addr);
  } else if (addr + access_size > g.beg + g.size) {
    if (addr < g.beg + g.size) addr = g.beg + g.size;
    str.append("%p is located %zd bytes to the right", (void *)addr,
               addr - (g.beg + g.size));
  } else {
    // Can it happen?
    str.append("%p is located %zd bytes inside", (void *)addr, addr - g.beg);
  }
  str.append(" of global variable '%s' defined in '",
             MaybeDemangleGlobalName(g.name));
  PrintGlobalLocation(&str, g);
  str.append("' (0x%zx) of size %zu\n", g.beg, g.size);
  str.append("%s", d.Default());
  PrintGlobalNameIfASCII(&str, g);
  Printf("%s", str.data());
}

bool GetGlobalAddressInformation(uptr addr, uptr access_size,
                                 GlobalAddressDescription *descr) {
  descr->addr = addr;
  int globals_num = GetGlobalsForAddress(addr, descr->globals, descr->reg_sites,
                                         ARRAY_SIZE(descr->globals));
  descr->size = globals_num;
  descr->access_size = access_size;
  return globals_num != 0;
}

bool DescribeAddressIfGlobal(uptr addr, uptr access_size,
                             const char *bug_type) {
  GlobalAddressDescription descr;
  if (!GetGlobalAddressInformation(addr, access_size, &descr)) return false;

  descr.Print(bug_type);
  return true;
}

void ShadowAddressDescription::Print() const {
  Printf("Address %p is located in the %s area.\n", addr, ShadowNames[kind]);
}

void GlobalAddressDescription::Print(const char *bug_type) const {
  for (int i = 0; i < size; i++) {
    DescribeAddressRelativeToGlobal(addr, access_size, globals[i]);
    if (bug_type &&
        0 == internal_strcmp(bug_type, "initialization-order-fiasco") &&
        reg_sites[i]) {
      Printf("  registered at:\n");
      StackDepotGet(reg_sites[i]).Print();
    }
  }
}

void StackAddressDescription::Print() const {
  Decorator d;
  char tname[128];
  Printf("%s", d.Location());
  Printf("Address %p is located in stack of thread T%d%s", addr, tid,
         ThreadNameWithParenthesis(tid, tname, sizeof(tname)));

  if (!frame_descr) {
    Printf("%s\n", d.Default());
    return;
  }
  Printf(" at offset %zu in frame%s\n", offset, d.Default());

  // Now we print the frame where the alloca has happened.
  // We print this frame as a stack trace with one element.
  // The symbolizer may print more than one frame if inlining was involved.
  // The frame numbers may be different than those in the stack trace printed
  // previously. That's unfortunate, but I have no better solution,
  // especially given that the alloca may be from entirely different place
  // (e.g. use-after-scope, or different thread's stack).
  Printf("%s", d.Default());
  StackTrace alloca_stack(&frame_pc, 1);
  alloca_stack.Print();

  InternalMmapVector<StackVarDescr> vars(16);
  if (!ParseFrameDescription(frame_descr, &vars)) {
    Printf(
        "AddressSanitizer can't parse the stack frame "
        "descriptor: |%s|\n",
        frame_descr);
    // 'addr' is a stack address, so return true even if we can't parse frame
    return;
  }
  uptr n_objects = vars.size();
  // Report the number of stack objects.
  Printf("  This frame has %zu object(s):\n", n_objects);

  // Report all objects in this frame.
  for (uptr i = 0; i < n_objects; i++) {
    uptr prev_var_end = i ? vars[i - 1].beg + vars[i - 1].size : 0;
    uptr next_var_beg = i + 1 < n_objects ? vars[i + 1].beg : ~(0UL);
    PrintAccessAndVarIntersection(vars[i], offset, access_size, prev_var_end,
                                  next_var_beg);
  }
  Printf(
      "HINT: this may be a false positive if your program uses "
      "some custom stack unwind mechanism or swapcontext\n");
  if (SANITIZER_WINDOWS)
    Printf("      (longjmp, SEH and C++ exceptions *are* supported)\n");
  else
    Printf("      (longjmp and C++ exceptions *are* supported)\n");

  DescribeThread(GetThreadContextByTidLocked(tid));
}

void HeapAddressDescription::Print() const {
  PrintHeapChunkAccess(addr, chunk_access);

  asanThreadRegistry().CheckLocked();
  AsanThreadContext *alloc_thread = GetThreadContextByTidLocked(alloc_tid);
  StackTrace alloc_stack = GetStackTraceFromId(alloc_stack_id);

  char tname[128];
  Decorator d;
  AsanThreadContext *free_thread = nullptr;
  if (free_tid != kInvalidTid) {
    free_thread = GetThreadContextByTidLocked(free_tid);
    Printf("%sfreed by thread T%d%s here:%s\n", d.Allocation(),
           free_thread->tid,
           ThreadNameWithParenthesis(free_thread, tname, sizeof(tname)),
           d.Default());
    StackTrace free_stack = GetStackTraceFromId(free_stack_id);
    free_stack.Print();
    Printf("%spreviously allocated by thread T%d%s here:%s\n", d.Allocation(),
           alloc_thread->tid,
           ThreadNameWithParenthesis(alloc_thread, tname, sizeof(tname)),
           d.Default());
  } else {
    Printf("%sallocated by thread T%d%s here:%s\n", d.Allocation(),
           alloc_thread->tid,
           ThreadNameWithParenthesis(alloc_thread, tname, sizeof(tname)),
           d.Default());
  }
  alloc_stack.Print();
  DescribeThread(GetCurrentThread());
  if (free_thread) DescribeThread(free_thread);
  DescribeThread(alloc_thread);
}

AddressDescription::AddressDescription(uptr addr, uptr access_size,
                                       bool shouldLockThreadRegistry) {
  if (GetShadowAddressInformation(addr, &data.shadow)) {
    data.kind = kAddressKindShadow;
    return;
  }
  if (GetHeapAddressInformation(addr, access_size, &data.heap)) {
    data.kind = kAddressKindHeap;
    return;
  }

  bool isStackMemory = false;
  if (shouldLockThreadRegistry) {
    ThreadRegistryLock l(&asanThreadRegistry());
    isStackMemory = GetStackAddressInformation(addr, access_size, &data.stack);
  } else {
    isStackMemory = GetStackAddressInformation(addr, access_size, &data.stack);
  }
  if (isStackMemory) {
    data.kind = kAddressKindStack;
    return;
  }

  if (GetGlobalAddressInformation(addr, access_size, &data.global)) {
    data.kind = kAddressKindGlobal;
    return;
  }
  data.kind = kAddressKindWild;
  addr = 0;
}

void PrintAddressDescription(uptr addr, uptr access_size,
                             const char *bug_type) {
  ShadowAddressDescription shadow_descr;
  if (GetShadowAddressInformation(addr, &shadow_descr)) {
    shadow_descr.Print();
    return;
  }

  GlobalAddressDescription global_descr;
  if (GetGlobalAddressInformation(addr, access_size, &global_descr)) {
    global_descr.Print(bug_type);
    return;
  }

  StackAddressDescription stack_descr;
  if (GetStackAddressInformation(addr, access_size, &stack_descr)) {
    stack_descr.Print();
    return;
  }

  HeapAddressDescription heap_descr;
  if (GetHeapAddressInformation(addr, access_size, &heap_descr)) {
    heap_descr.Print();
    return;
  }

  // We exhausted our possibilities. Bail out.
  Printf(
      "AddressSanitizer can not describe address in more detail "
      "(wild memory access suspected).\n");
}
}  // namespace __asan
