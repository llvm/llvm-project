//===-- asan_interceptors.cc ----------------------------------------------===//
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
// Interceptors for operators new and delete.
//===----------------------------------------------------------------------===//

#include "asan_allocator.h"
#include "asan_internal.h"
#include "asan_stack.h"

#include "interception/interception.h"

#include <stddef.h>

// C++ operators can't have dllexport attributes on Windows. We export them
// anyway by passing extra -export flags to the linker, which is exactly that
// dllexport would normally do. We need to export them in order to make the
// VS2015 dynamic CRT (MD) work.
#if SANITIZER_WINDOWS
#define CXX_OPERATOR_ATTRIBUTE
#define COMMENT_EXPORT(sym) __pragma(comment(linker, "/export:"##sym))
#ifdef _WIN64
COMMENT_EXPORT("??2@YAPEAX_K@Z")                     // operator new
COMMENT_EXPORT("??2@YAPEAX_KAEBUnothrow_t@std@@@Z")  // operator new nothrow
COMMENT_EXPORT("??3@YAXPEAX@Z")                      // operator delete
COMMENT_EXPORT("??3@YAXPEAX_K@Z")                    // sized operator delete
COMMENT_EXPORT("??_U@YAPEAX_K@Z")                    // operator new[]
COMMENT_EXPORT("??_V@YAXPEAX@Z")                     // operator delete[]
#else
COMMENT_EXPORT("??2@YAPAXI@Z")                    // operator new
COMMENT_EXPORT("??2@YAPAXIABUnothrow_t@std@@@Z")  // operator new nothrow
COMMENT_EXPORT("??3@YAXPAX@Z")                    // operator delete
COMMENT_EXPORT("??3@YAXPAXI@Z")                   // sized operator delete
COMMENT_EXPORT("??_U@YAPAXI@Z")                   // operator new[]
COMMENT_EXPORT("??_V@YAXPAX@Z")                   // operator delete[]
#endif
#undef COMMENT_EXPORT
#else
#define CXX_OPERATOR_ATTRIBUTE INTERCEPTOR_ATTRIBUTE
#endif

using namespace __asan;  // NOLINT

// FreeBSD prior v9.2 have wrong definition of 'size_t'.
// http://svnweb.freebsd.org/base?view=revision&revision=232261
#if SANITIZER_FREEBSD && SANITIZER_WORDSIZE == 32
#include <sys/param.h>
#if __FreeBSD_version <= 902001  // v9.2
#define size_t unsigned
#endif  // __FreeBSD_version
#endif  // SANITIZER_FREEBSD && SANITIZER_WORDSIZE == 32

// This code has issues on OSX.
// See https://github.com/google/sanitizers/issues/131.

// Fake std::nothrow_t and std::align_val_t to avoid including <new>.
namespace std {
struct nothrow_t {};
enum class align_val_t: size_t {};
}  // namespace std

// TODO(alekseys): throw std::bad_alloc instead of dying on OOM.
#define OPERATOR_NEW_BODY(type, nothrow) \
  GET_STACK_TRACE_MALLOC;\
  void *res = asan_memalign(0, size, &stack, type);\
  if (!nothrow && UNLIKELY(!res)) DieOnFailure::OnOOM();\
  return res;
#define OPERATOR_NEW_BODY_ALIGN(type, nothrow) \
  GET_STACK_TRACE_MALLOC;\
  void *res = asan_memalign((uptr)align, size, &stack, type);\
  if (!nothrow && UNLIKELY(!res)) DieOnFailure::OnOOM();\
  return res;

// On OS X it's not enough to just provide our own 'operator new' and
// 'operator delete' implementations, because they're going to be in the
// runtime dylib, and the main executable will depend on both the runtime
// dylib and libstdc++, each of those'll have its implementation of new and
// delete.
// To make sure that C++ allocation/deallocation operators are overridden on
// OS X we need to intercept them using their mangled names.
#if !SANITIZER_MAC
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size)
{ OPERATOR_NEW_BODY(FROM_NEW, false /*nothrow*/); }
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size)
{ OPERATOR_NEW_BODY(FROM_NEW_BR, false /*nothrow*/); }
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size, std::nothrow_t const&)
{ OPERATOR_NEW_BODY(FROM_NEW, true /*nothrow*/); }
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size, std::nothrow_t const&)
{ OPERATOR_NEW_BODY(FROM_NEW_BR, true /*nothrow*/); }
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size, std::align_val_t align)
{ OPERATOR_NEW_BODY_ALIGN(FROM_NEW, false /*nothrow*/); }
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size, std::align_val_t align)
{ OPERATOR_NEW_BODY_ALIGN(FROM_NEW_BR, false /*nothrow*/); }
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size, std::align_val_t align, std::nothrow_t const&)
{ OPERATOR_NEW_BODY_ALIGN(FROM_NEW, true /*nothrow*/); }
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size, std::align_val_t align, std::nothrow_t const&)
{ OPERATOR_NEW_BODY_ALIGN(FROM_NEW_BR, true /*nothrow*/); }

#else  // SANITIZER_MAC
INTERCEPTOR(void *, _Znwm, size_t size) {
  OPERATOR_NEW_BODY(FROM_NEW, false /*nothrow*/);
}
INTERCEPTOR(void *, _Znam, size_t size) {
  OPERATOR_NEW_BODY(FROM_NEW_BR, false /*nothrow*/);
}
INTERCEPTOR(void *, _ZnwmRKSt9nothrow_t, size_t size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY(FROM_NEW, true /*nothrow*/);
}
INTERCEPTOR(void *, _ZnamRKSt9nothrow_t, size_t size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY(FROM_NEW_BR, true /*nothrow*/);
}
#endif

#define OPERATOR_DELETE_BODY(type) \
  GET_STACK_TRACE_FREE;\
  asan_free(ptr, &stack, type);

#if !SANITIZER_MAC
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr) NOEXCEPT {
  OPERATOR_DELETE_BODY(FROM_NEW);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr) NOEXCEPT {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(FROM_NEW);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, size_t size) NOEXCEPT {
  GET_STACK_TRACE_FREE;
  asan_sized_free(ptr, size, &stack, FROM_NEW);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, size_t size) NOEXCEPT {
  GET_STACK_TRACE_FREE;
  asan_sized_free(ptr, size, &stack, FROM_NEW_BR);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, std::align_val_t) NOEXCEPT {
  OPERATOR_DELETE_BODY(FROM_NEW);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, std::align_val_t) NOEXCEPT {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, std::align_val_t, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(FROM_NEW);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, std::align_val_t, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, size_t size, std::align_val_t) NOEXCEPT {
  GET_STACK_TRACE_FREE;
  asan_sized_free(ptr, size, &stack, FROM_NEW);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, size_t size, std::align_val_t) NOEXCEPT {
  GET_STACK_TRACE_FREE;
  asan_sized_free(ptr, size, &stack, FROM_NEW_BR);
}

#else  // SANITIZER_MAC
INTERCEPTOR(void, _ZdlPv, void *ptr) {
  OPERATOR_DELETE_BODY(FROM_NEW);
}
INTERCEPTOR(void, _ZdaPv, void *ptr) {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}
INTERCEPTOR(void, _ZdlPvRKSt9nothrow_t, void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(FROM_NEW);
}
INTERCEPTOR(void, _ZdaPvRKSt9nothrow_t, void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}
#endif
