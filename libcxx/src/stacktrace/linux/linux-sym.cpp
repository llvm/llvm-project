//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/config.h"

#if defined(_LIBCPP_STACKTRACE_LINUX)

#  include "linux.h"

#  include <cassert>
#  include <dlfcn.h>
#  include <unistd.h>

#  include <__stacktrace/context.h>
#  include <__stacktrace/entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

bool symbolize_entry(entry& entry) {
  bool ret = false;
  Dl_info info;
  if (dladdr((void*)entry.__addr_, &info)) {
    ret = true; // at least partially successful
    if (info.dli_fname && entry.__file_.empty()) {
      // provide at least the binary filename in case we cannot lookup source location
      entry.__file_ = info.dli_fname;
    }
    if (info.dli_sname && entry.__desc_.empty()) {
      // provide at least the mangled name; try to unmangle in a later step
      entry.__desc_ = info.dli_sname;
    }
  }
  return ret;
}

// NOTE:  We can use `dlfcn` to resolve addresses to symbols, which works great --
// except for symbols in the main program.  If addr2line-style tools are enabled, that step
// might also be able to get symbols directly from the binary's debug info.
void linux::symbolize() {
  for (auto& entry : cx_.__entries_) {
    symbolize_entry(entry);
  }
  // Symbols might be missing, because both (1) Linux's `dladdr` won't try to resolve non-exported symbols,
  // which can be the case for the main program executable; and (2) debug info was not preserved.
  // As a last resort, this function (see `linux-elf.cpp`) can still access symbol table directly.
  image* mainELF = images::get().mainProg();
  if (mainELF && !mainELF->name_.empty()) {
    resolve_main_elf_syms(mainELF->name_);
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_LINUX
