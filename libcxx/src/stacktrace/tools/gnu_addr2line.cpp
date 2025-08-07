//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>

#if __has_include(<spawn.h>) && _LIBCPP_STACKTRACE_ALLOW_TOOLS_AT_RUNTIME

#  include <__stacktrace/basic_stacktrace.h>
#  include <__stacktrace/memory.h>
#  include <__stacktrace/stacktrace_entry.h>
#  include <cctype>
#  include <cerrno>
#  include <csignal>
#  include <cstddef>
#  include <cstdlib>
#  include <spawn.h>
#  include <sys/fcntl.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>

#  include "stacktrace/tools/tools.h"
#  include <__stacktrace/images.h>

// clang-format off

// XXX addr2line only supports one input file to resolve addresses for;
// XXX should invoke once for each program image we get in our stacktrace?

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

bool addr2line::build_argv() {
  auto* main_image = images().main_prog_image();
  _LIBCPP_ASSERT(main_image, "could not determine main program image");
  _LIBCPP_ASSERT(!main_image->name_.empty(), "could not determine main program image name");
  if (!(main_image && !main_image->name_.empty())) {
    return false;
  }
  push_arg("/usr/bin/env");
  push_arg(tool_prog_);
  push_arg("--functions");
  push_arg("--demangle");
  push_arg("--basenames");
  push_arg("-e");
  push_arg(main_image->name_);
  auto* it  = base_.entries_begin();
  auto* end = base_.entries_end();
  while (it != end) {
    auto& entry = *(entry_base*)(it++);
    push_arg("%p", (void*)entry.adjusted_addr());
  }
  return true;
}

/*
Example:
--
addr2line \
  --functions --demangle --basenames \
  -e $BUILDDIR/libcxx/test/libcxx/stacktrace/Output/use_available_progs.pass.cpp.dir/t.tmp.exe \
  0x000100000610 0x000100000618 0x000100000620

NOTE: might not demangle even if we ask for `--demangle`
NOTE: currently seeing a malloc double-free in homebrew (macos) binutils 2.45 build of addr2line
      (which we ignore)

Output: (2 lines per input address)
---
Z5func0v
use_available_progs.pass.cpp:78
Z5func1v
use_available_progs.pass.cpp:81
Z5func2v
use_available_progs.pass.cpp:84
*/

void addr2line::parse_sym(entry_base& entry, std::string_view view) const {
  if (!view.starts_with("??")) {
    // XXX should check for "_Z" prefix (mangled symbol) and use cxxabi.h / demangle?
    entry.__desc_ = view;
  }
}

void addr2line::parse_loc(entry_base& entry, std::string_view view) const {
  if (!view.starts_with("??")) {
    auto colon = view.find_last_of(":");
    if (colon != string_view::npos) {
      entry.__file_ = view.substr(0, colon);
      entry.__line_ = atoi(view.data() + colon + 1);
    }
  }
}

template struct _LIBCPP_EXPORTED_FROM_ABI __executable_name<addr2line>;
template bool _LIBCPP_EXPORTED_FROM_ABI __has_working_executable<addr2line>();

template<> bool _LIBCPP_EXPORTED_FROM_ABI __run_tool<addr2line>(base& base, arena& arena) {
  addr2line tool{base, arena};
  if (!tool.build_argv()) { return false; }
  spawner spawner{tool, base};
  if (spawner.errno_) { return false; }

  str line                    ;               // our read buffer
  auto* entry_iter = base.entries_begin();    // position at first entry
  while (spawner.stream_.good()) {            // loop until we get EOF from tool stdout
    std::string_view view;

    std::getline(spawner.stream_, line);      // consume one line
    view = tool_base::strip(line);            // remove trailing and leading whitespace
    if (view.empty()) { continue; }           // blank line: restart loop, checking for EOF
    tool.parse_sym(*entry_iter, view);        // expecting symbol name

    std::getline(spawner.stream_, line);      // consume one line
    view = tool_base::strip(line);            // remove trailing and leading whitespace
    if (view.empty()) { continue; }           // blank line: restart loop, checking for EOF
    tool.parse_loc(*entry_iter, view);        // expecting "/path/to/sourcefile.cpp:42"

    ++entry_iter;                             // one entry per two lines
  }

  return true;
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
