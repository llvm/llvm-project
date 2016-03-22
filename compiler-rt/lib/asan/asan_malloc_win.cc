//===-- asan_malloc_win.cc ------------------------------------------------===//
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
// Windows-specific malloc interception.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_WINDOWS

#include "asan_allocator.h"
#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_stack.h"
#include "interception/interception.h"

#include <stddef.h>

using namespace __asan;  // NOLINT

// MT: Simply defining functions with the same signature in *.obj
// files overrides the standard functions in the CRT.
// MD: Memory allocation functions are defined in the CRT .dll,
// so we have to intercept them before they are called for the first time.

#if ASAN_DYNAMIC
# define ALLOCATION_FUNCTION_ATTRIBUTE
#else
# define ALLOCATION_FUNCTION_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE
#endif

extern "C" {
ALLOCATION_FUNCTION_ATTRIBUTE
void free(void *ptr) {
  GET_STACK_TRACE_FREE;
  return asan_free(ptr, &stack, FROM_MALLOC);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void _free_dbg(void *ptr, int) {
  free(ptr);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void _free_base(void *ptr) {
  free(ptr);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void cfree(void *ptr) {
  CHECK(!"cfree() should not be used on Windows");
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *malloc(size_t size) {
  GET_STACK_TRACE_MALLOC;
  return asan_malloc(size, &stack);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *_malloc_base(size_t size) {
  return malloc(size);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *_malloc_dbg(size_t size, int, const char *, int) {
  return malloc(size);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *calloc(size_t nmemb, size_t size) {
  GET_STACK_TRACE_MALLOC;
  return asan_calloc(nmemb, size, &stack);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *_calloc_base(size_t nmemb, size_t size) {
  return calloc(nmemb, size);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *_calloc_dbg(size_t nmemb, size_t size, int, const char *, int) {
  return calloc(nmemb, size);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *_calloc_impl(size_t nmemb, size_t size, int *errno_tmp) {
  return calloc(nmemb, size);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *realloc(void *ptr, size_t size) {
  GET_STACK_TRACE_MALLOC;
  return asan_realloc(ptr, size, &stack);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *_realloc_dbg(void *ptr, size_t size, int) {
  CHECK(!"_realloc_dbg should not exist!");
  return 0;
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *_realloc_base(void *ptr, size_t size) {
  return realloc(ptr, size);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *_recalloc(void *p, size_t n, size_t elem_size) {
  if (!p)
    return calloc(n, elem_size);
  const size_t size = n * elem_size;
  if (elem_size != 0 && size / elem_size != n)
    return 0;
  return realloc(p, size);
}

ALLOCATION_FUNCTION_ATTRIBUTE
size_t _msize(void *ptr) {
  GET_CURRENT_PC_BP_SP;
  (void)sp;
  return asan_malloc_usable_size(ptr, pc, bp);
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *_expand(void *memblock, size_t size) {
  // _expand is used in realloc-like functions to resize the buffer if possible.
  // We don't want memory to stand still while resizing buffers, so return 0.
  return 0;
}

ALLOCATION_FUNCTION_ATTRIBUTE
void *_expand_dbg(void *memblock, size_t size) {
  return _expand(memblock, size);
}

// TODO(timurrrr): Might want to add support for _aligned_* allocation
// functions to detect a bit more bugs.  Those functions seem to wrap malloc().

int _CrtDbgReport(int, const char*, int,
                  const char*, const char*, ...) {
  ShowStatsAndAbort();
}

int _CrtDbgReportW(int reportType, const wchar_t*, int,
                   const wchar_t*, const wchar_t*, ...) {
  ShowStatsAndAbort();
}

int _CrtSetReportMode(int, int) {
  return 0;
}
}  // extern "C"

namespace __asan {
void ReplaceSystemMalloc() {
#if defined(ASAN_DYNAMIC)
  // We don't check the result because CRT might not be used in the process.
  __interception::OverrideFunction("free", (uptr)free);
  __interception::OverrideFunction("malloc", (uptr)malloc);
  __interception::OverrideFunction("_malloc_crt", (uptr)malloc);
  __interception::OverrideFunction("calloc", (uptr)calloc);
  __interception::OverrideFunction("_calloc_crt", (uptr)calloc);
  __interception::OverrideFunction("realloc", (uptr)realloc);
  __interception::OverrideFunction("_realloc_crt", (uptr)realloc);
  __interception::OverrideFunction("_recalloc", (uptr)_recalloc);
  __interception::OverrideFunction("_recalloc_crt", (uptr)_recalloc);
  __interception::OverrideFunction("_msize", (uptr)_msize);
  __interception::OverrideFunction("_expand", (uptr)_expand);

  // Override different versions of 'operator new' and 'operator delete'.
  // No need to override the nothrow versions as they just wrap the throw
  // versions.
  // FIXME: Unfortunately, MSVC miscompiles the statements that take the
  // addresses of the array versions of these operators,
  // see https://connect.microsoft.com/VisualStudio/feedbackdetail/view/946992
  // We might want to try to work around this by [inline] assembly or compiling
  // parts of the RTL with Clang.
  void *(*op_new)(size_t sz) = operator new;
  void (*op_delete)(void *p) = operator delete;
  void *(*op_array_new)(size_t sz) = operator new[];
  void (*op_array_delete)(void *p) = operator delete[];
  __interception::OverrideFunction("??2@YAPAXI@Z", (uptr)op_new);
  __interception::OverrideFunction("??3@YAXPAX@Z", (uptr)op_delete);
  __interception::OverrideFunction("??_U@YAPAXI@Z", (uptr)op_array_new);
  __interception::OverrideFunction("??_V@YAXPAX@Z", (uptr)op_array_delete);
#endif
}
}  // namespace __asan

#endif  // _WIN32
