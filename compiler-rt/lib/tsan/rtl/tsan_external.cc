//===-- tsan_external.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_rtl.h"

namespace __tsan {

#define CALLERPC ((uptr)__builtin_return_address(0))

const uptr kMaxTag = 128;  // Limited to 65,536, since MBlock only stores tags
                           // as 16-bit values, see tsan_defs.h.

const char *registered_tags[kMaxTag];
static atomic_uint32_t used_tags{1};  // Tag 0 means "no tag". NOLINT

const char *GetObjectTypeFromTag(uptr tag) {
  if (tag == 0) return nullptr;
  // Invalid/corrupted tag?  Better return NULL and let the caller deal with it.
  if (tag >= atomic_load(&used_tags, memory_order_relaxed)) return nullptr;
  return registered_tags[tag];
}

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void *__tsan_external_register_tag(const char *object_type) {
  uptr new_tag = atomic_fetch_add(&used_tags, 1, memory_order_relaxed);
  CHECK_LT(new_tag, kMaxTag);
  registered_tags[new_tag] = internal_strdup(object_type);
  return (void *)new_tag;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_external_assign_tag(void *addr, void *tag) {
  CHECK_LT(tag, atomic_load(&used_tags, memory_order_relaxed));
  Allocator *a = allocator();
  MBlock *b = nullptr;
  if (a->PointerIsMine((void *)addr)) {
    void *block_begin = a->GetBlockBegin((void *)addr);
    if (block_begin) b = ctx->metamap.GetBlock((uptr)block_begin);
  }
  if (b) {
    b->tag = (uptr)tag;
  }
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_external_read(void *addr, void *caller_pc, void *tag) {
  CHECK_LT(tag, atomic_load(&used_tags, memory_order_relaxed));
  ThreadState *thr = cur_thread();
  thr->external_tag = (uptr)tag;
  FuncEntry(thr, (uptr)caller_pc);
  MemoryRead(thr, CALLERPC, (uptr)addr, kSizeLog8);
  FuncExit(thr);
  thr->external_tag = 0;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_external_write(void *addr, void *caller_pc, void *tag) {
  CHECK_LT(tag, atomic_load(&used_tags, memory_order_relaxed));
  ThreadState *thr = cur_thread();
  thr->external_tag = (uptr)tag;
  FuncEntry(thr, (uptr)caller_pc);
  MemoryWrite(thr, CALLERPC, (uptr)addr, kSizeLog8);
  FuncExit(thr);
  thr->external_tag = 0;
}
}  // extern "C"

}  // namespace __tsan
