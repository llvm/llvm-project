//===-- tsan_new_delete.cc ----------------------------------------------===//
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
// Interceptors for operators new and delete.
//===----------------------------------------------------------------------===//
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "tsan_interceptors.h"

using namespace __tsan;  // NOLINT

namespace std {
struct nothrow_t {};
}  // namespace std

DECLARE_REAL(void *, malloc, uptr size)
DECLARE_REAL(void, free, void *ptr)

#define OPERATOR_NEW_BODY(mangled_name) \
  if (cur_thread()->in_symbolizer) \
    return InternalAlloc(size); \
  void *p = 0; \
  {  \
    SCOPED_INTERCEPTOR_RAW(mangled_name, size); \
    p = user_alloc(thr, pc, size); \
  }  \
  invoke_malloc_hook(p, size);  \
  return p;

SANITIZER_INTERFACE_ATTRIBUTE
void *operator new(__sanitizer::uptr size);
void *operator new(__sanitizer::uptr size) {
  OPERATOR_NEW_BODY(_Znwm);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *operator new[](__sanitizer::uptr size);
void *operator new[](__sanitizer::uptr size) {
  OPERATOR_NEW_BODY(_Znam);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *operator new(__sanitizer::uptr size, std::nothrow_t const&);
void *operator new(__sanitizer::uptr size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY(_ZnwmRKSt9nothrow_t);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *operator new[](__sanitizer::uptr size, std::nothrow_t const&);
void *operator new[](__sanitizer::uptr size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY(_ZnamRKSt9nothrow_t);
}

#define OPERATOR_DELETE_BODY(mangled_name) \
  if (ptr == 0) return;  \
  if (cur_thread()->in_symbolizer) \
    return InternalFree(ptr); \
  invoke_free_hook(ptr);  \
  SCOPED_INTERCEPTOR_RAW(mangled_name, ptr);  \
  user_free(thr, pc, ptr);

SANITIZER_INTERFACE_ATTRIBUTE
void operator delete(void *ptr) NOEXCEPT;
void operator delete(void *ptr) NOEXCEPT {
  OPERATOR_DELETE_BODY(_ZdlPv);
}

SANITIZER_INTERFACE_ATTRIBUTE
void operator delete[](void *ptr) NOEXCEPT;
void operator delete[](void *ptr) NOEXCEPT {
  OPERATOR_DELETE_BODY(_ZdaPv);
}

SANITIZER_INTERFACE_ATTRIBUTE
void operator delete(void *ptr, std::nothrow_t const&);
void operator delete(void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(_ZdlPvRKSt9nothrow_t);
}

SANITIZER_INTERFACE_ATTRIBUTE
void operator delete[](void *ptr, std::nothrow_t const&);
void operator delete[](void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(_ZdaPvRKSt9nothrow_t);
}
