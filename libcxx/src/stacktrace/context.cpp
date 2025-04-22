//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__config_site>

#include <list>

#include <__stacktrace/basic_stacktrace.h>
#include <__stacktrace/stacktrace_entry.h>

#include <__stacktrace/alloc.h>
#include <__stacktrace/context.h>
#include <__stacktrace/entry.h>
#include <__stacktrace/to_string.h>

#include "common/config.h"

#if defined(_LIBCPP_STACKTRACE_LINUX)
#  include "linux/linux.h"
#endif

#if defined(_LIBCPP_STACKTRACE_APPLE)
#  include "osx/osx.h"
#endif

#if defined(_LIBCPP_STACKTRACE_CAN_SPAWN_TOOLS)
#  include "tools/pspawn.h"
#endif

#if defined(_LIBCPP_STACKTRACE_COLLECT_UNWIND)
#  include "unwind/unwind.h"
#endif

#if defined(_LIBCPP_STACKTRACE_USE_DBGHELP)
#  include "windows/win_impl.h"
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_HIDDEN void context::do_stacktrace(size_t skip, size_t max_depth) {
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
#if defined(_LIBCPP_STACKTRACE_APPLE)
  osx osx{*this};
  auto& mod_ident  = osx;
  auto& symbolizer = osx;
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

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD
