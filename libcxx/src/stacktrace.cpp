//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;

#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

module std;

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __stacktrace_impl {

_LIBCPP_EXPORTED_FROM_ABI string __format_entry_description(uintptr_t __addr) {
  if (__addr == 0)
    return string("0x0");

#if defined(_WIN32)
  MEMORY_BASIC_INFORMATION __mbi;
  if (VirtualQuery(reinterpret_cast<LPCVOID>(__addr), &__mbi, sizeof(__mbi))) {
    HMODULE __h_mod = reinterpret_cast<HMODULE>(__mbi.AllocationBase);
    char __mod_path[MAX_PATH];

    if (__h_mod && GetModuleFileNameA(__h_mod, __mod_path, sizeof(__mod_path))) {
      const char* __file_name = __mod_path;
      for (const char* __p = __mod_path; *__p; ++__p) {
        if (*__p == '\\' || *__p == '/') {
          __file_name = __p + 1;
        }
      }

      uintptr_t __rva = __addr - reinterpret_cast<uintptr_t>(__h_mod);

      char __out_buf[256];
      int __len = std::snprintf(
          __out_buf, sizeof(__out_buf), "%s!+0x%llx", __file_name, static_cast<unsigned long long>(__rva));
      if (__len > 0)
        return string(__out_buf, static_cast<size_t>(__len));
    }
  }
#endif

  char __buf[32];
  int __len = std::snprintf(__buf, sizeof(__buf), "0x%llx", static_cast<unsigned long long>(__addr));
  if (__len > 0)
    return string(__buf, static_cast<size_t>(__len));

  return string("0x0");
}

} // namespace __stacktrace_impl

_LIBCPP_END_NAMESPACE_STD
