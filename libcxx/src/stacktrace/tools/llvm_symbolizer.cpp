//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__config_site>
#include <cstddef>
#include <iostream>
#include <string_view>

#include "../common/debug.h"
#include "tools.h"

#include <__stacktrace/context.h>
#include <__stacktrace/entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

// TODO(stacktrace23): possible to link against `libLLVMSymbolize.a`, or some shared obj at runtime (does that exist?)

std::pmr::list<std::pmr::string> llvm_symbolizer::buildArgs(context& cx) const {
  auto& alloc = cx.__alloc_;
  auto ret    = alloc.new_string_list();
  ret.push_back(progName_);
  ret.push_back("--demangle");
  ret.push_back("--no-inlines");
  ret.push_back("--verbose");
  ret.push_back("--relativenames");
  ret.push_back("--functions=short");
  for (auto& entry : cx.__entries_) {
    auto addr_string = alloc.hex_string(entry.__addr_unslid_);
    debug() << "@@@ " << addr_string << " " << entry.__file_ << " " << entry.__file_.empty() << '\n';
    if (!entry.__file_.empty()) {
      auto arg = alloc.new_string(entry.__file_.size() + 40);
      arg += "FILE:";
      arg += entry.__file_;
      arg += " ";
      arg += addr_string;
      ret.push_back(arg);
    } else {
      ret.push_back(addr_string);
    }
  }
  return ret;
}

void llvm_symbolizer::parseOutput(context& cx, entry& entry, std::istream& output) const {
  // clang-format off
/*
With "--verbose", parsing is a little easier, or at least, more reliable;
probably the best solution (until we have a JSON parser).
Example output, verbatim, between the '---' lines:
---
test1<test_alloc<std::__1::stacktrace_entry> >
  Filename: /data/code/llvm-project/libcxx/test/std/diagnostics/stacktrace/basic.cons.pass.cpp
  Function start filename: /data/code/llvm-project/libcxx/test/std/diagnostics/stacktrace/basic.cons.pass.cpp
  Function start line: 114
  Function start address: 0x8dd0
  Line: 116
  Column: 14

---
Note that this includes an extra empty line as a terminator.
*/
  // clang-format on

  auto& alloc = cx.__alloc_;
  auto line   = alloc.new_string(256);
  std::string_view tmp;
  while (true) {
    std::getline(output, line);
    while (isspace(line.back())) {
      line.pop_back();
    }
    if (line.empty()) {
      return;
    }
    if (!line.starts_with("  ")) {
      // The symbol has no leading whitespace, while the other
      // lines with "fields" like line, column, filename, etc.
      // start with two spaces.
      if (line != "??") {
        entry.__desc_ = line;
      }
    } else if (line.starts_with("  Filename:")) {
      tmp = line;
      tmp = tmp.substr(tmp.find_first_of(":") + 2); // skip ": "
      if (tmp != "??") {
        entry.__file_ = tmp;
      }
    } else if (line.starts_with("  Line:")) {
      tmp = line;
      tmp = tmp.substr(tmp.find_first_of(":") + 2); // skip ": "
      if (tmp != "??" && tmp != "0") {
        uint32_t lineno = 0;
        auto pos        = 0;
        while (isdigit(tmp[pos])) {
          lineno = lineno * 10 + (tmp[pos++] - '0');
        }
        entry.__line_ = lineno;
      }
    }
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD
