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
#  include <unistd.h>

#  include <experimental/__stacktrace/detail/context.h>
#  include <experimental/__stacktrace/detail/entry.h>

#  include "../common/fd.h"
#  include "elf.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

/**
When trying to collect a stacktrace under Linux, there are narrow (but still quite common) cases where we will fail
to resolve symbols.  Linux's `dl` doesn't want to read symbols from the non-exported symbol table at runtime,
and older versions of `addr2line` or `llvm-symbolizer` will also not resolve these.

This implementation the minimum necessary to resolve symbols.  It can identify this as an ELF (32 or 64 bits), locate
the symbol and symbol-string table, and fill in any remaining missing symbols.
*/
void linux::resolve_main_elf_syms(std::string_view main_elf_name) {
  // We can statically initialize these, because main_elf_name should be the same every time.
  static fd_mmap _mm(main_elf_name);
  if (_mm) {
    static elf::ELF _this_elf(_mm.addr_);
    if (_this_elf) {
      for (auto& entry : cx_.__entries_) {
        if (entry.__desc_.empty() && entry.__file_ == main_elf_name) {
          entry.__desc_ = _this_elf.getSym(entry.__addr_unslid_).name();
        }
      }
    }
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_LINUX
