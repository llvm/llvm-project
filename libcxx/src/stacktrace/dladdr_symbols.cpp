//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// `dladdr` resolves an address to the *nearest* preceding symbol in its image's dynamic symbol
// table. That table only contains symbols with external linkage, so this is inherently
// best-effort: TU-local (`static`) functions, and any image built without a dynamic symbol
// table at all, simply won't resolve. That's fine -- callers treat an empty `__desc_` as "not
// found", never as an error.
#if !defined(_WIN32) && !defined(_AIX) && __has_include(<dlfcn.h>)

#  include <__stacktrace/stacktrace_entry.h>
#  include <cstdlib>
#  include <cxxabi.h>
#  include <dlfcn.h>

#  include "symbols.h"

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

namespace __stacktrace {

namespace {

// Best-effort Itanium demangle: falls back to the raw symbol name if demangling fails,
// e.g. a plain C symbol, or a name the demangler doesn't recognize.
std::string demangle(const char* __name) {
  int __status      = 0;
  char* __demangled = abi::__cxa_demangle(__name, nullptr, nullptr, &__status);
  if (__status != 0 || !__demangled) {
    return __name;
  }
  std::string __ret(__demangled);
  free(__demangled);
  return __ret;
}

} // namespace

void __populate_symbols(_Context& __cx) {
  for (_Entry& entry : __cx.__entry_iters_(__cx.__self_)) {
    Dl_info __info{};
    if (dladdr(reinterpret_cast<void*>(entry.__addr_), &__info) && __info.dli_sname) {
      entry.__desc_ = demangle(__info.dli_sname);
    }
  }
}

} // namespace __stacktrace

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_WIN32) && !defined(_AIX) && __has_include(<dlfcn.h>)
