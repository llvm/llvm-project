//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__config_site>

#include <__stacktrace/base.h>

#include "stacktrace/linux/impl.h"
#include "stacktrace/macos/impl.h"
#include "stacktrace/tools/tools.h"
#include "stacktrace/unwind/impl.h"
#include "stacktrace/windows/impl.h"

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __stacktrace {

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void builder::build_stacktrace(size_t skip, size_t max_depth) {
  // First get the instruction addresses, populate __entries_
  win_impl dbghelp{*this};
  unwind unwind{*this};
  dbghelp.collect(skip + 1, max_depth);
  unwind.collect(skip + 1, max_depth);

  // (Can't proceed if empty)
  if (!__entries_.size()) {
    return;
  }

  // Associate addrs with binaries (ELF/MachO/etc.)
  macos macos{*this};
  linux linux{*this};
  dbghelp.ident_modules();
  macos.ident_modules();
  linux.ident_modules();

  // Resolve addresses to symbols, filename, linenumber
  spawner pspawn{*this};
  dbghelp.resolve_lines();
  pspawn.resolve_lines();

  // Populate missing symbols, if any.
  dbghelp.symbolize();
  macos.symbolize();
  linux.symbolize();
}

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD
