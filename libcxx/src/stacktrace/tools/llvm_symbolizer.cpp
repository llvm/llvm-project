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
#  include <__stacktrace/images.h>
#  include <__stacktrace/memory.h>
#  include <__stacktrace/stacktrace_entry.h>
#  include <cstddef>
#  include <cstdlib>
#  include <unistd.h>

#  include "stacktrace/tools/tools.h"

// clang-format off

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

bool llvm_symbolizer::build_argv() {
  push_arg("/usr/bin/env");
  push_arg(tool_prog_);
  push_arg("--demangle");
  push_arg("--no-inlines");
  push_arg("--verbose");
  push_arg("--relativenames");
  push_arg("--functions=short");
  auto* it  = base_.entries_begin();
  auto* end = base_.entries_end();
  while (it != end) {
    auto& entry = *(entry_base*)(it++);
    if (entry.__image_ && !entry.__image_->name_.empty()) {
      fixed_str<PATH_MAX> image_path = entry.__image_->name_;
      push_arg("FILE:%s %p", image_path.data(), (void*)entry.adjusted_addr());
    } else {
      push_arg("%p", (void*)entry.adjusted_addr());
    }
  }
  return true;
}

void llvm_symbolizer::parse(entry_base** entry_iter, std::string_view view) const {
  /*
  Parsing is most reliable with `--verbose` option (short of having a JSON parser).  Example:

  test1<test_alloc<std::__1::stackbuilder_entry> >
    Filename: /data/code/llvm-project/libcxx/test/std/diagnostics/stacktrace/basic.cons.pass.cpp
    Function start filename: /data/code/llvm-project/libcxx/test/std/diagnostics/stacktrace/basic.cons.pass.cpp
    Function start line: 114
    Function start address: 0x8dd0
    Line: 116
    Column: 14
  */

  if (!view.starts_with("  ")) { // line without leading whitespace starts a new entry
    ++*entry_iter;               // advance to next entry
    _LIBCPP_ASSERT(*entry_iter >= base_.entries_begin(), "out of range");
    _LIBCPP_ASSERT(*entry_iter < base_.entries_end(), "out of range");
    auto& entry = **entry_iter;
    if (view != "??") { entry.__desc_ = view; }

  } else if (view.starts_with("  Filename:")) {
    auto& entry = **entry_iter;
    auto tmp    = view.substr(view.find_first_of(":") + 2); // skip ": "
    if (tmp != "??") { entry.__file_ = tmp; }

  } else if (view.starts_with("  Line:")) {
    auto& entry = **entry_iter;
    auto tmp    = view;
    tmp         = tmp.substr(tmp.find_first_of(":") + 2); // skip ": "
    if (tmp != "??" && tmp != "0") { entry.__line_ = atoi(tmp.data()); }
  }
}

template struct _LIBCPP_EXPORTED_FROM_ABI __executable_name<llvm_symbolizer>;
template bool _LIBCPP_EXPORTED_FROM_ABI __has_working_executable<llvm_symbolizer>();

template<> bool _LIBCPP_EXPORTED_FROM_ABI __run_tool<llvm_symbolizer>(base& base, arena& arena) {
  llvm_symbolizer tool{base, arena};
  if (!tool.build_argv()) { return false; }
  spawner spawner{tool, base};
  if (spawner.errno_) { return false; }

  fixed_str<PATH_MAX * 2> line;                 // our read buffer
  auto* entry_iter = base.entries_begin() - 1;  // "before first" entry
  while (spawner.stream_.good()) {              // loop until we get EOF from tool stdout
    std::getline(spawner.stream_, line);        // consume a line from stdout
    auto view = tool_base::rstrip(line);        // remove trailing (but not leading) whitespace
    if (tool_base::rstrip(view).empty()) { continue; }  // skip if line had nothing, or _only_ whitespace
    tool.parse(&entry_iter, view);              // send to parser (who might update entry_iter)
  }

  return true;
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
