//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
/*
Self-contained WinAPI import surface for the runtime (no <windows.h>).
*/
#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma comment(lib, "ntdll.lib")
#endif

namespace std::error_domains::__herbceptions_detail {

namespace win32 {

extern "C" {
// ntdll
__declspec(dllimport) void *__stdcall
RtlAllocateHeap(void *heap, ::std::uint_least32_t flags,
                ::std::size_t size) noexcept;
__declspec(dllimport) int __stdcall
RtlFreeHeap(void *heap, ::std::uint_least32_t flags, void *ptr) noexcept;
__declspec(dllimport) void *__stdcall RtlPcToFileHeader(void const *pc,
                                                        void **base) noexcept;

// kernel32
__declspec(dllimport) void *__stdcall GetProcessHeap() noexcept;
__declspec(dllimport) void *__stdcall HeapAlloc(void *heap,
                                                ::std::uint_least32_t flags,
                                                ::std::size_t bytes) noexcept;
__declspec(dllimport) int __stdcall
HeapFree(void *heap, ::std::uint_least32_t flags, void *ptr) noexcept;
__declspec(dllimport) void *__stdcall LocalFree(void *mem) noexcept;
__declspec(dllimport) void *__stdcall
GetModuleHandleA(char const *name) noexcept;
__declspec(dllimport) void *__stdcall
GetModuleHandleW(wchar_t const *name) noexcept;
__declspec(dllimport) ::std::uint_least32_t __stdcall
FormatMessageA(::std::uint_least32_t flags, void const *source,
               ::std::uint_least32_t message_id,
               ::std::uint_least32_t language_id, char *buffer,
               ::std::uint_least32_t size, char *arguments) noexcept;
__declspec(dllimport) ::std::uint_least32_t __stdcall
FormatMessageW(::std::uint_least32_t flags, void const *source,
               ::std::uint_least32_t message_id,
               ::std::uint_least32_t language_id, wchar_t *buffer,
               ::std::uint_least32_t size, char *arguments) noexcept;
}

} // namespace win32

// Minimal NT type the runtime touches: EXCEPTION_RECORD.
struct exception_record {
  ::std::uint_least32_t ExceptionCode;
  ::std::uint_least32_t ExceptionFlags;
  void *ExceptionRecord;
  void *ExceptionAddress;
  ::std::uint_least32_t NumberParameters;
  ::std::uintptr_t ExceptionInformation[15];
};

using EXCEPTION_RECORD = exception_record;

} // namespace std::error_domains::__herbceptions_detail
