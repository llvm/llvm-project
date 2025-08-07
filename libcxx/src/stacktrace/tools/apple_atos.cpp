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
#  include <cstddef>
#  include <cstdlib>
#  include <unistd.h>

#  include "stacktrace/tools/tools.h"

// clang-format off

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

bool atos::build_argv() {
  push_arg("/usr/bin/env");
  push_arg(tool_prog_);
  push_arg("-p");
  push_arg("%d", getpid());
  auto* it  = base_.entries_begin();
  auto* end = base_.entries_end();
  while (it != end) {
    auto& entry = *(entry_base*)(it++);
    push_arg("%p", (void*)entry.__addr_);
  }
  return true;
}

void atos::parse(entry_base& entry, std::string_view view) const {
  // With debug info we should get everything we need in one line:
  // main (in testprog) (/Users/steve/code/notes/testprog.cc:208)

  // view:       main (in t.tmp.exe) (simple.o0.nosplit.pass.cpp:19)
  // advance i to:   ^
  size_t i = 0;
  while (i < view.size() && !isspace(view[i])) { ++i; }
  entry.__desc_ = view.substr(0, i);
  view = lstrip(ldrop(view, i));

  // view:       (in t.tmp.exe) (simple.o0.nosplit.pass.cpp:19)
  // looking for:             ^^^
  auto pos = view.find(") (");
  if (pos == std::string_view::npos) { return; }
  view = ldrop(view, pos + 3);    // simple.o0.nosplit.pass.cpp:19)
  view = drop_suffix(view, ")");  // simple.o0.nosplit.pass.cpp:19
  pos = view.find_last_of(":");   //                           ^here
  if (pos == std::string_view::npos) { return; }
  entry.__file_ = view.substr(0, pos);
  auto lineno = view.substr(pos + 1);
  entry.__line_ = lineno.empty() ? 0 : stoi(string(lineno));
}

template struct _LIBCPP_EXPORTED_FROM_ABI __executable_name<atos>;
template bool _LIBCPP_EXPORTED_FROM_ABI __has_working_executable<atos>();

template<> bool _LIBCPP_EXPORTED_FROM_ABI __run_tool<atos>(base& base, arena& arena) {
  atos tool{base, arena};
  if (!tool.build_argv()) { return false; }
  spawner spawner{tool, base};
  if (spawner.errno_) { return false; }

  str line;                                   // our read buffer
  auto* entry_iter = base.entries_begin();    // position at first entry
  while (spawner.stream_.good()) {            // loop until we get EOF from tool stdout
    std::getline(spawner.stream_, line);      // consume a line from stdout
    auto view = tool_base::strip(line);       // remove trailing and leading whitespace
    if (view.empty()) { continue; }           // skip blank lines
    tool.parse(*entry_iter, view);
    ++entry_iter;                             // one line per entry
  }

  return true;
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
