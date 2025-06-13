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

// Explicitly instantiate this template class
template class std::allocator<std::stacktrace_entry>;

namespace __stacktrace {

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void builder::build_stacktrace(size_t skip, size_t max_depth) {
#if defined(_LIBCPP_WIN32API)
  // Windows implementation
  win_impl dbghelp{*this};
  dbghelp.collect(skip + 1, max_depth);
  if (!__entries_.size()) {
    return;
  }
  dbghelp.ident_modules();
  dbghelp.resolve_lines();
  dbghelp.symbolize();

#else
  // Non-Windows; assume Unix-like.

  // For spawning `addr2line` or similar external process
  spawner pspawn{*this};

  // `Unwind.h` or `libunwind.h` often available on Linux/OSX etc.
  unwind unwind{*this};
  unwind.collect(skip + 1, max_depth);
  if (!__entries_.size()) {
    return;
  }

#  if defined(__APPLE__)
  // Specific to MacOS and other Apple SDKs
  macos macos{*this};
  macos.ident_modules();
  pspawn.resolve_lines();
  macos.symbolize();

#  else
  // Linux and other other platforms
  linux linux{*this};
  linux.ident_modules();
  pspawn.resolve_lines();
  linux.symbolize();

#  endif
#endif
}

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD
