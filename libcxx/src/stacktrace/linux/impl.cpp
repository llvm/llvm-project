//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "stacktrace/linux/impl.h"
#include "stacktrace/config.h"

#if defined(__linux__)

#  include <cassert>
#  include <dlfcn.h>
#  include <link.h>
#  include <stacktrace>
#  include <unistd.h>

#  include "stacktrace/config.h"
#  include "stacktrace/utils/fd.h"
#  include "stacktrace/utils/image.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

void linux::ident_modules() {
  auto& images = images::get();

  // Aside from the left/right sentinels in the array (hence the 2),
  // are there any other real images?
  if (images.count_ <= 2) {
    return;
  }

  auto mainProg = images.mainProg();
  if (mainProg) {
    builder_.__main_prog_path_ = mainProg->name_;
  }

  unsigned index = 1; // Starts at one, and is moved around in this loop
  for (auto& entry : builder_.__entries_) {
    while (images[index].loaded_at_ > entry.__addr_actual_) {
      --index;
    }
    while (images[index + 1].loaded_at_ <= entry.__addr_actual_) {
      ++index;
    }
    entry.__addr_unslid_ = entry.__addr_actual_ - images[index].slide_;
    entry.__file_        = builder_.__alloc_.make_str(images[index].name_);
  }
}

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
      for (auto& entry : builder_.__entries_) {
        if (entry.__desc_->empty() && entry.__file_ == main_elf_name) {
          auto name     = _this_elf.getSym(entry.__addr_unslid_).name();
          entry.__desc_ = builder_.__alloc_.make_str(name);
        }
      }
    }
  }
}

bool symbolize_entry(alloc& alloc, entry_base& entry) {
  bool ret = false;
  Dl_info info;
  if (dladdr((void*)entry.__addr_actual_, &info)) {
    ret = true; // at least partially successful
    if (info.dli_fname && entry.__file_->empty()) {
      // provide at least the binary filename in case we cannot lookup source location
      entry.__file_ = alloc.make_str(info.dli_fname);
    }
    if (info.dli_sname && entry.__desc_->empty()) {
      // provide at least the mangled name; try to unmangle in a later step
      entry.__desc_ = alloc.make_str(info.dli_sname);
    }
  }
  return ret;
}

// NOTE:  We can use `dlfcn` to resolve addresses to symbols, which works great --
// except for symbols in the main program.  If addr2line-style tools are enabled, that step
// might also be able to get symbols directly from the binary's debug info.
void linux::symbolize() {
  for (auto& entry : builder_.__entries_) {
    symbolize_entry(builder_.__alloc_, entry);
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

#endif // __linux__
