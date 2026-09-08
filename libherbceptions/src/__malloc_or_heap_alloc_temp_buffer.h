//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdlib>
#ifdef _WIN32
#include "win32_imports.h"
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

/*
Two heap flavors live side by side:

  - The NT flavor goes straight to ntdll's RtlAllocateHeap/RtlFreeHeap with
    the process heap handle read out of the PEB through the TEB ("register"
    part), so no kernel32 import is needed for it. Used by the exception
    domains, which only ever run on NT-family kernels. The declarations
    mirror fast_io_core_impl/allocation/nt_preliminary_definition.h.
  - The win32 flavor uses kernel32's HeapAlloc/HeapFree, which exists since
    Windows 95; used by the win32/com message machinery and by the itanium
    domain so they keep working everywhere.

Every imported function is declared noexcept so no exception semantics can
bite across the C ABI boundary.
*/
namespace std::error_domains::__herbceptions_detail {

#ifdef _WIN32

#if defined(_MSC_VER) && !defined(_KERNEL_MODE) && !defined(_WIN32_WINDOWS)
#pragma comment(lib, "ntdll.lib")
#endif

inline void *__process_heap() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
  // TEB -> PEB (gs:[0x60]); PEB->ProcessHeap lives at 0x30.
#if defined(_MSC_VER) && !defined(__clang__)
  void *const peb{reinterpret_cast<void *>(__readgsqword(0x60))};
#else
  void *peb;
  __asm__("movq %%gs:0x60, %0" : "=r"(peb));
#endif
  return *reinterpret_cast<void **>(reinterpret_cast<unsigned char *>(peb) +
                                    0x30);
#elif defined(_M_IX86) || defined(__i386__)
  // TEB -> PEB (fs:[0x30]); PEB->ProcessHeap lives at 0x18.
#if defined(_MSC_VER) && !defined(__clang__)
  void *const peb{reinterpret_cast<void *>(__readfsdword(0x30))};
#else
  void *peb;
  __asm__("movl %%fs:0x30, %0" : "=r"(peb));
#endif
  return *reinterpret_cast<void **>(reinterpret_cast<unsigned char *>(peb) +
                                    0x18);
#else
  return win32::GetProcessHeap();
#endif
}

// NT flavor: ntdll heap on the PEB process heap.
inline void *__nt_heap_alloc_or_die(::std::size_t __sz) noexcept {
  void *__bufferptr = win32::RtlAllocateHeap(__process_heap(), 0, __sz);
  if (__bufferptr == nullptr)
    ::std::abort();
  return __bufferptr;
}
inline void __nt_heap_free(void *__bufferptr) noexcept {
  if (__bufferptr == nullptr)
    return;
  win32::RtlFreeHeap(__process_heap(), 0, __bufferptr);
}

// Win32 flavor: kernel32 heap, available on every Windows including 9x.
inline void *__win32_heap_alloc_or_die(::std::size_t __sz) noexcept {
  void *__bufferptr = win32::HeapAlloc(win32::GetProcessHeap(), 0, __sz);
  if (__bufferptr == nullptr)
    ::std::abort();
  return __bufferptr;
}
inline void __win32_heap_free(void *__bufferptr) noexcept {
  if (__bufferptr == nullptr)
    return;
  win32::HeapFree(win32::GetProcessHeap(), 0, __bufferptr);
}

#endif

// Dispatchers used by the runtime code: the nt flavor for the exception
// domains, the win32 flavor elsewhere.
inline void *__malloc_or_heap_alloc_or_die(::std::size_t __sz) noexcept {
#ifdef _WIN32
  return __nt_heap_alloc_or_die(__sz);
#else
  void *__bufferptr = ::std::malloc(__sz);
  if (__bufferptr == nullptr)
    ::std::abort();
  return __bufferptr;
#endif
}
inline void __free_or_heap_dealloc(void *__bufferptr) noexcept {
#ifdef _WIN32
  __nt_heap_free(__bufferptr);
#else
  if (__bufferptr != nullptr) {
    ::std::free(__bufferptr);
  }
#endif
}
inline void *__malloc_or_die(::std::size_t __sz) noexcept {
  void *__bufferptr = ::std::malloc(__sz);
  if (__bufferptr == nullptr)
    ::std::abort();
  return __bufferptr;
}

template <unsigned __malloconly = 0>
class __basic_malloc_or_heapalloc_temp_buffer {
public:
  void *__bufferptr{};
  constexpr __basic_malloc_or_heapalloc_temp_buffer() noexcept = default;
  constexpr __basic_malloc_or_heapalloc_temp_buffer(void *__bp) noexcept
      : __bufferptr(__bp) {}
  __basic_malloc_or_heapalloc_temp_buffer(
      __basic_malloc_or_heapalloc_temp_buffer const &) = delete;
  __basic_malloc_or_heapalloc_temp_buffer &
  operator=(__basic_malloc_or_heapalloc_temp_buffer const &) = delete;
  ~__basic_malloc_or_heapalloc_temp_buffer() {
    if (this->__bufferptr == nullptr)
      return;
#ifdef _WIN32
    if constexpr (__malloconly == 2) {
      win32::LocalFree(this->__bufferptr);
    } else
#endif
        if constexpr (__malloconly == 1) {
      ::std::free(this->__bufferptr);
    }
#ifdef _WIN32
    else if constexpr (__malloconly == 3) {
      __win32_heap_free(this->__bufferptr);
    }
#endif
    else {
      __free_or_heap_dealloc(this->__bufferptr);
    }
  }
};

using __malloc_or_heapalloc_temp_buffer =
    __basic_malloc_or_heapalloc_temp_buffer<0>;
using __local_free_temp_buffer = __basic_malloc_or_heapalloc_temp_buffer<
#ifdef _WIN32
    2
#else
    0
#endif
    >;
using __malloc_temp_buffer = __basic_malloc_or_heapalloc_temp_buffer<
#ifdef _WIN32
    1
#else
    0
#endif
    >;
using __heapalloc_temp_buffer = __basic_malloc_or_heapalloc_temp_buffer<
#ifdef _WIN32
    3
#else
    0
#endif
    >;

} // namespace std::error_domains::__herbceptions_detail
