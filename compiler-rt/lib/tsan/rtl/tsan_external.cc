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
#include "tsan_interceptors.h"

namespace __tsan {

#define CALLERPC ((uptr)__builtin_return_address(0))

const char *registered_tags[kExternalTagMax];
static atomic_uint32_t used_tags{kExternalTagFirstUserAvailable};  // NOLINT.

const char *GetObjectTypeFromTag(uptr tag) {
  if (tag == 0) return nullptr;
  // Invalid/corrupted tag?  Better return NULL and let the caller deal with it.
  if (tag >= atomic_load(&used_tags, memory_order_relaxed)) return nullptr;
  return registered_tags[tag];
}

void InsertShadowStackFrameForTag(ThreadState *thr, uptr tag) {
  FuncEntry(thr, (uptr)&registered_tags[tag]);
}

uptr TagFromShadowStackFrame(uptr pc) {
  uptr tag_count = atomic_load(&used_tags, memory_order_relaxed);
  void *pc_ptr = (void *)pc;
  if (pc_ptr < &registered_tags[0] || pc_ptr >= &registered_tags[tag_count])
    return 0;
  return (const char **)pc_ptr - &registered_tags[0];
}

#if !SANITIZER_GO

typedef void(*AccessFunc)(ThreadState *, uptr, uptr, int);
void ExternalAccess(void *addr, void *caller_pc, void *tag, AccessFunc access) {
  CHECK_LT(tag, atomic_load(&used_tags, memory_order_relaxed));
  ThreadState *thr = cur_thread();
  if (caller_pc) FuncEntry(thr, (uptr)caller_pc);
  InsertShadowStackFrameForTag(thr, (uptr)tag);
  bool in_ignored_lib;
  if (!caller_pc || !libignore()->IsIgnored((uptr)caller_pc, &in_ignored_lib)) {
    access(thr, CALLERPC, (uptr)addr, kSizeLog1);
  }
  FuncExit(thr);
  if (caller_pc) FuncExit(thr);
}

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void *__tsan_external_register_tag(const char *object_type) {
  uptr new_tag = atomic_fetch_add(&used_tags, 1, memory_order_relaxed);
  CHECK_LT(new_tag, kExternalTagMax);
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
  ExternalAccess(addr, caller_pc, tag, MemoryRead);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_external_write(void *addr, void *caller_pc, void *tag) {
  ExternalAccess(addr, caller_pc, tag, MemoryWrite);
}
}  // extern "C"

#endif  // !SANITIZER_GO

}  // namespace __tsan
