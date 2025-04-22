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

#include "tools.h"

#include <__stacktrace/context.h>
#include <__stacktrace/entry.h>
#include <string>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

std::pmr::list<std::pmr::string> addr2line::buildArgs(context& cx) const {
  auto& alloc = cx.__alloc_;
  auto ret    = alloc.new_string_list();
  if (cx.__main_prog_path_.empty()) {
    // Should not have reached here but be graceful anyway
    ret.push_back("/bin/false");
    return ret;
  }

  ret.push_back(progName_);
  ret.push_back("--functions");
  ret.push_back("--demangle");
  ret.push_back("--basenames");
  ret.push_back("--pretty-print"); // This "human-readable form" is easier to parse
  ret.push_back("-e");
  ret.push_back(cx.__main_prog_path_);
  for (auto& entry : cx.__entries_) {
    ret.push_back(alloc.hex_string(entry.__addr_unslid_));
  }
  return ret;
}

void addr2line::parseOutput(context& trace, entry& entry, std::istream& output) const {
  // clang-format off
/*
Example:
--
llvm-addr2line -e foo --functions --demangle --basenames --pretty-print --no-inlines 0x11a0 0x1120 0x3d58 0x1284

Output: (1 line per input address)
--
main at foo.cc:15
register_tm_clones at crtstuff.c:0
GCC_except_table2 at foo.cc:0
test::Foo::Foo(int) at foo.cc:11
*/
  // clang-format on

  std::pmr::string line{&trace.__alloc_};
  std::getline(output, line);
  while (isspace(line.back())) {
    line.pop_back();
  }
  if (line.empty()) {
    return;
  }
  // Split at the sequence " at ".  Barring weird symbols
  // having " at " in them, this should work.
  auto sepIndex = line.find(" at ");
  if (sepIndex == std::string::npos) {
    return;
  }
  if (sepIndex > 0) {
    entry.__desc_ = line.substr(0, sepIndex);
  }
  auto fileBegin = sepIndex + 4;
  if (fileBegin >= line.size()) {
    return;
  }
  auto fileline = line.substr(fileBegin);
  auto colon    = fileline.find_last_of(":");
  if (colon > 0 && !fileline.starts_with("?")) {
    entry.__file_ = fileline.substr(0, colon);
  }

  if (colon == std::string::npos) {
    return;
  }
  uint32_t lineno = 0;
  auto pos        = colon;
  while (isdigit(fileline[++pos])) {
    lineno = lineno * 10 + (fileline[pos] - '0');
  }
  entry.__line_ = lineno;
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD
