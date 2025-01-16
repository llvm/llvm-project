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
#include <istream>
#include <string>
#include <unistd.h>

#include "tools.h"

#include <experimental/__stacktrace/detail/alloc.h>
#include <experimental/__stacktrace/detail/context.h>
#include <experimental/__stacktrace/detail/entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

std::pmr::list<std::pmr::string> atos::buildArgs(context& cx) const {
  auto& alloc = cx.__alloc_;
  auto ret    = alloc.new_string_list();
  ret.push_back(progName_);
  ret.push_back("-p");
  ret.push_back(alloc.u64_string(getpid()));
  // TODO(stackcx23): Allow options in env, e.g. LIBCPP_STACKTRACE_OPTIONS=FullPath
  // ret.push_back("--fullPath");
  for (auto& entry : cx.__entries_) {
    ret.push_back(alloc.hex_string(entry.__addr_));
  }
  return ret;
}

void atos::parseOutput(context& cx, entry& entry, std::istream& output) const {
  // Simple example:
  //
  //   main (in testprog) (/Users/steve/code/notes/testprog.cc:208)
  //
  // Assuming this is always atos's format (except when it returns empty lines)
  // we can split the string like so:
  //
  //   main (in testprog) (/Users/steve/code/notes/testprog.cc:208)
  //   ^^^^-----^^^^^^^^---^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-^^^-
  //   sym      module     filename                            line
  //
  // Note that very strange filenames or module names can confuse this.
  // We'll do the best we can for a decent result, while definitely ensuring safety
  // (i.e. careful with our bound-checking).
  //
  // Another more interesting example (with an added newline for legibility):
  //
  //   std::__1::basic_ios<char, std::__1::char_traits<char>>::fill[abi:ne190107]() const (in testprog)
  //   (/opt/homebrew/Cellar/llvm/19.1.7_1/bin/../include/c++/v1/ios:0
  //
  // If this more or less fits our expected format we'll take these data,
  // even if the line number is 0.

  auto line = cx.__alloc_.new_string(256);
  std::getline(output, line);
  while (isspace(line.back())) {
    line.pop_back();
  }
  if (line.empty()) {
    return;
  }
  auto buf  = line.data();
  auto size = line.size();

  auto* end    = buf + size;
  auto* symEnd = strstr(buf, " (in ");
  if (!symEnd) {
    return;
  }
  auto* modBegin = symEnd + 5;
  auto* modEnd   = strstr(modBegin, ") (");
  if (!modEnd) {
    return;
  }
  auto* fileBegin = modEnd + 3; // filename starts just after that
  if (fileBegin >= end) {
    return;
  }
  auto const* lastColon = fileBegin; // we'll search for last colon after filename
  char const* nextColon;
  while ((nextColon = strstr(lastColon + 1, ":"))) { // skip colons in filename (e.g. in "C:\foo.cpp")
    lastColon = nextColon;
  }

  std::string_view sym{buf, size_t(symEnd - buf)};
  // In case a previous step could not obtain the symbol name,
  // we have the name provided by atos; only use that if we have no symbol
  // (no need to copy more strings otherwise).
  if (entry.__desc_.empty() && !sym.empty()) {
    entry.__desc_ = sym;
  }

  std::string_view file{fileBegin, size_t(lastColon - fileBegin)};
  if (file != "?" && file != "??" && !file.empty()) {
    entry.__file_ = file;
  }

  unsigned lineno = 0;
  for (auto* digit = lastColon + 1; digit < end && isdigit(*digit); ++digit) {
    lineno = (lineno * 10) + unsigned(*digit - '0');
  }
  entry.__line_ = lineno;
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD
