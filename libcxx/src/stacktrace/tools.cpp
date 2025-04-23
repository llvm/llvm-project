//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "stacktrace/config.h"

#if defined(_LIBCPP_STACKTRACE_CAN_SPAWN_TOOLS)

#  include <__config>
#  include <__config_site>
#  include <cassert>
#  include <cerrno>
#  include <csignal>
#  include <cstddef>
#  include <cstdlib>
#  include <spawn.h>
#  include <sys/fcntl.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>

#  include "stacktrace/context.h"
#  include "stacktrace/tools.h"
#  include "stacktrace/utils.h"
#  include <__stacktrace/basic_stacktrace.h>
#  include <__stacktrace/stacktrace_entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

namespace {

/**
Returns a list of `tool` objects that can be tried during the first stacktrace operation.
The result is an array of `tool*` pointers with a final `nullptr` terminator.

The returned tool array pointer is saved during `resolve_lines`'s static init,
with the first working tool being used from that point forward.

This will first add any tools specified by these defines (in order):
- LIBCXX_STACKTRACE_FORCE_LLVM_SYMBOLIZER_PATH
- LIBCXX_STACKTRACE_FORCE_GNU_ADDR2LINE_PATH
- LIBCXX_STACKTRACE_FORCE_APPLE_ATOS_PATH

Then any tools specified by the environment variables of the same names (and added in the same order).

Finally, this adds the tools `llvm-symbolizer`, `addr2line`, and `atos`, in that order.
These tools won't have absolute paths, so $PATH will be searched for these.

Called by `findWorkingTool` during its static initialization, therefore this is used in a threadsafe way.
*/
tool const** toolList() {
  constexpr static size_t kMax = 20;
  static tool const* array_[kMax];
  size_t count = 0;

  auto add = [&](tool* t) {
    assert(count < kMax - 1);
    array_[count++] = t;
  };

#  define STRINGIFY0(x) #x
#  define STRINGIFY(x) STRINGIFY0(x)

#  if defined(LIBCXX_STACKTRACE_FORCE_LLVM_SYMBOLIZER_PATH)
  {
    static llvm_symbolizer t{STRINGIFY(LIBCXX_STACKTRACE_FORCE_LLVM_SYMBOLIZER_PATH)};
    add(&t);
  }
#  endif
#  if defined(LIBCXX_STACKTRACE_FORCE_GNU_ADDR2LINE_PATH)
  {
    static addr2line t{STRINGIFY(LIBCXX_STACKTRACE_FORCE_GNU_ADDR2LINE_PATH)};
    add(&t);
  }
#  endif
#  if defined(LIBCXX_STACKTRACE_FORCE_APPLE_ATOS_PATH)
  {
    static atos t{STRINGIFY(LIBCXX_STACKTRACE_FORCE_APPLE_ATOS_PATH)};
    add(&t);
  }
#  endif

  if (getenv("LIBCXX_STACKTRACE_FORCE_LLVM_SYMBOLIZER_PATH")) {
    static llvm_symbolizer t{getenv("LIBCXX_STACKTRACE_FORCE_LLVM_SYMBOLIZER_PATH")};
    add(&t);
  }
  if (getenv("LIBCXX_STACKTRACE_FORCE_GNU_ADDR2LINE_PATH")) {
    static addr2line t{getenv("LIBCXX_STACKTRACE_FORCE_GNU_ADDR2LINE_PATH")};
    add(&t);
  }
  if (getenv("LIBCXX_STACKTRACE_FORCE_APPLE_ATOS_PATH")) {
    static atos t{getenv("LIBCXX_STACKTRACE_FORCE_APPLE_ATOS_PATH")};
    add(&t);
  }

  {
    static llvm_symbolizer t;
    add(&t);
  }
  {
    static addr2line t;
    add(&t);
  }
  {
    static atos t;
    add(&t);
  }

  array_[count] = nullptr; // terminator
  return array_;
}

// Try a handful of different addr2line-esque tools, returning the first discovered, or else nullptr
tool const* findWorkingTool(auto& trace) {
  // Try each of these programs and stop at the first one that works.
  // There's no synchronization so this is subject to races on setting `prog`,
  // but as long as one thread ends up setting it to something that works, we're good.
  // Programs to try ($PATH is searched):
  auto* it = toolList();
  tool const* prog;
  while ((prog = *it++)) {
    pspawn test{*prog}; // Just try to run with "--help"
    try {
      std::pmr::list<std::pmr::string> testArgs{&trace.__alloc_};
      testArgs.push_back(prog->progName_);
      testArgs.push_back("--help");
      test.fa_.redirectInNull();
      test.fa_.redirectOutNull();
      test.fa_.redirectErrNull();
      test.spawn(testArgs);
      if (test.wait() == 0) {
        // Success
        return prog;
      }
    } catch (failed const&) {
      /* ignore during probe attempt */
    }
  }
  return nullptr;
}

} // namespace

void spawner::resolve_lines() {
  // The address-to-line tool that worked (after lazy-setting, below)
  static tool const* prog{nullptr};
  // Whether we should not attempt because tool detection previously failed.
  static bool fail{false};

  // If this previously failed, don't try again.
  if (fail) {
    return;
  }

  if (!prog) {
    prog = findWorkingTool(cx_);
    if (!prog) {
      fail = true;
      return;
    }
  }
  assert(prog);

  char buf[256];
  pspawn_tool proc(*prog, cx_, buf, sizeof(buf));
  try {
    proc.run();
  } catch (failed const& failed) {
    debug() << failed.what();
    if (failed.errno_) {
      debug() << " (" << failed.errno_ << " " << strerror(failed.errno_) << ')';
    }
    debug() << '\n';
  }
}

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

#endif
