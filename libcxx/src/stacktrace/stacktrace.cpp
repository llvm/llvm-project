//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__config_site>

#include <iomanip>
#include <ios>
#include <iostream>
#include <sstream>
#include <stacktrace>
#include <string>

#include "stacktrace/config.h"
#include "stacktrace/context.h"
#include "stacktrace/utils.h"

#if defined(_LIBCPP_STACKTRACE_LINUX)
#  include "stacktrace/linux.h"
#endif

#if defined(_LIBCPP_STACKTRACE_MACOS)
#  include "stacktrace/macos.h"
#endif

#if defined(_LIBCPP_STACKTRACE_CAN_SPAWN_TOOLS)
#  include "stacktrace/tools.h"
#endif

#if defined(_LIBCPP_STACKTRACE_COLLECT_UNWIND)
#  include "stacktrace/unwind.h"
#endif

#if defined(_LIBCPP_STACKTRACE_USE_DBGHELP)
#  include "stacktrace/windows.h"
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// `to_string`-related non-member functions

_LIBCPP_EXPORTED_FROM_ABI string to_string(const stacktrace_entry& __entry) {
  return __stacktrace::__to_string()(__entry);
}

_LIBCPP_EXPORTED_FROM_ABI ostream& operator<<(ostream& __os, const stacktrace_entry& __entry) {
  __stacktrace::__to_string()(__os, __entry);
  return __os;
}

namespace __stacktrace {

/*
 * `to_string` Helpers
 */

_LIBCPP_EXPORTED_FROM_ABI void __to_string::operator()(ostream& __os, std::stacktrace_entry const& entry) {
  // Although 64-bit addresses are 16 nibbles long, they're often <= 0x7fff_ffff_ffff
  constexpr static int __k_addr_width = (sizeof(void*) > 4) ? 12 : 8;

  __os << "0x" << std::hex << std::setfill('0') << std::setw(__k_addr_width) << entry.native_handle();
  if (!entry.description().empty()) {
    __os << ": " << entry.description();
  }
  if (!entry.source_file().empty()) {
    __os << ": " << entry.source_file();
  }
  if (entry.source_line()) {
    __os << ":" << std::dec << entry.source_line();
  }
}

_LIBCPP_EXPORTED_FROM_ABI void
__to_string::operator()(ostream& __os, std::stacktrace_entry const* __entries, size_t __count) {
  /*
   * Print each entry as a line, as per `operator()`, with additional whitespace
   * at the start of the line, and only a newline added at the end:
   *
   *   frame   1: 0xbeefbeefbeef: _symbol_name: /path/to/file.cc:123
   */
  if (!__count) {
    __os << "(empty stacktrace)";
  } else {
    for (size_t __i = 0; __i < __count; __i++) {
      if (__i) {
        // Insert newlines between entries (but not before the first or after the last)
        __os << std::endl;
      }
      __os << "  frame " << std::setw(3) << std::setfill(' ') << std::dec << (__i + 1) << ": ";
      (*this)(__os, __entries[__i]);
    }
  }
}

_LIBCPP_EXPORTED_FROM_ABI string __to_string::operator()(std::stacktrace_entry const& entry) {
  stringstream __ss;
  (*this)(__ss, entry);
  return __ss.str();
}

_LIBCPP_EXPORTED_FROM_ABI string __to_string::operator()(std::stacktrace_entry const* __entries, size_t __count) {
  stringstream __ss;
  (*this)(__ss, __entries, __count);
  return __ss.str();
}

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void
__impl(size_t skip,
       size_t max_depth,
       alloc& alloc,
       std::function<void(size_t)> resize_func,
       std::function<void(size_t, std::stacktrace_entry&&)> assign_func) {
  context cx{alloc};
  cx.do_stacktrace(1 + skip, max_depth);
  resize_func(cx.__entries_.size());
  size_t i = 0;
  for (auto& entry : cx.__entries_) {
    assign_func(i++, entry);
  }
}

entry::operator std::stacktrace_entry() {
  std::stacktrace_entry __ret;
  __ret.__addr_ = __addr_;
  __ret.__desc_ = __desc_;
  __ret.__file_ = __file_;
  __ret.__line_ = __line_;
  return __ret;
}

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void context::do_stacktrace(size_t skip, size_t max_depth) {
  /*
  Here we declare stacktrace components or "backends" which will handle the different tasks:

  (1) get the addresses from the call stack
  (2) identify program images in process virtual space (program binary, plus modules, shared/dynamic libs)
  (3) resolve using debug info, and possibly with an external tool on the $PATH
  (4+) extra passes to get symbols, in case 3 couldn't

  Based on the macros produced by `stacktrace.h`, throw all backends we have available at the task.  Ideally the #ifdef
  gauntlet below should result in one of each of the above functions: (1) collector, (2) mod_ident, (3) resolver, (4)
  symbolizer.  If any are missing or duplicated that is still fine; we work with zero or all the available utilities.

  All these classes do their best to provide any of the requested fields they can: (symbol, filename, source line),
  substituting if needed with something reasonable.  For example, if the source filename and line are not available
  then we will at least report that the address and symbol are in the module `foo.exe`.

  These components should also tolerate: missing data, weirdly-formatted data (e.g. from the external tools), or even
  already-populated data.  We take care not to crash / abort / throw in any of these, and we'll silently fail.  See
  `common/debug.h` for a debugging logger you can enable at runtime.
  */

#if defined(_LIBCPP_STACKTRACE_USE_DBGHELP)
  win_impl dbghelp{*this};
  auto& collector  = dbghelp;
  auto& mod_ident  = dbghelp;
  auto& resolver   = dbghelp;
  auto& symbolizer = dbghelp;
#endif
#if defined(_LIBCPP_STACKTRACE_COLLECT_UNWIND)
  unwind unwind{*this};
  auto& collector = unwind;
#endif
#if defined(_LIBCPP_STACKTRACE_MACOS)
  macos macos{*this};
  auto& mod_ident  = macos;
  auto& symbolizer = macos;
#endif
#if defined(_LIBCPP_STACKTRACE_LINUX)
  linux linux{*this};
  auto& mod_ident  = linux;
  auto& symbolizer = linux;
#endif
#if defined(_LIBCPP_STACKTRACE_CAN_SPAWN_TOOLS)
  spawner pspawn{*this};
  auto& resolver = pspawn;
#endif

  collector.collect(skip + 1, max_depth); // First get the instruction addresses, populate __entries_
  if (__entries_.size()) {                // (Can't proceed if empty)
    mod_ident.ident_modules();            // Associate addrs with binaries (ELF/MachO/etc.)
    resolver.resolve_lines();             // Resolve addresses to symbols, filename, linenumber
    symbolizer.symbolize();               // Populate missing symbols, if any.
  }
}

int fd_streambuf::underflow() {
  int bytesRead = ::read(fd_, buf_, size_);
  if (bytesRead < 0) {
    throw failed("I/O error reading from child process", errno);
  }
  if (bytesRead == 0) {
    return traits_type::eof();
  }
  setg(buf_, buf_, buf_ + bytesRead);
  return int(*buf_);
}

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD
