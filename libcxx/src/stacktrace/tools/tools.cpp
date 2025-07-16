//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__config_site>

#if __has_include(<spawn.h>) && _LIBCPP_STACKTRACE_ALLOW_TOOLS_AT_RUNTIME

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

#  include <__stacktrace/base.h>
#  include <__stacktrace/basic.h>
#  include <__stacktrace/entry.h>

#  include "stacktrace/tools/tools.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

namespace {

_LIBCPP_HIDE_FROM_ABI base::str hex_string(base& base, uintptr_t __addr) {
  char __ret[19]; // "0x" + 16 digits + NUL
  auto __size = snprintf(__ret, sizeof(__ret), "0x%016llx", (unsigned long long)__addr);
  return base.make_str(__ret, size_t(__size));
}

_LIBCPP_HIDE_FROM_ABI base::str u64_string(base& base, uintptr_t __val) {
  char __ret[21]; // 20 digits max + NUL
  auto __size = snprintf(__ret, sizeof(__ret), "%zu", __val);
  return base.make_str(__ret, size_t(__size));
}

#  define STRINGIFY0(x) #x
#  define STRINGIFY(x) STRINGIFY0(x)

bool try_tools(base& base, function<bool(tool const&)> cb) {
  char const* prog_name;

  if ((prog_name = getenv("LIBCXX_STACKTRACE_FORCE_LLVM_SYMBOLIZER_PATH"))) {
    if (cb(llvm_symbolizer{base, prog_name})) {
      return true;
    }
  } else {
#  if defined(LIBCXX_STACKTRACE_FORCE_LLVM_SYMBOLIZER_PATH)
    if (cb(llvm_symbolizer{base, STRINGIFY(LIBCXX_STACKTRACE_FORCE_LLVM_SYMBOLIZER_PATH)})) {
      return true;
    }
#  else
    if (cb(llvm_symbolizer{base})) {
      return true;
    }
#  endif
  }

  if ((prog_name = getenv("LIBCXX_STACKTRACE_FORCE_GNU_ADDR2LINE_PATH"))) {
    if (cb(addr2line{base, prog_name})) {
      return true;
    }
  } else {
#  if defined(LIBCXX_STACKTRACE_FORCE_GNU_ADDR2LINE_PATH)
    if (cb(addr2line{base, STRINGIFY(LIBCXX_STACKTRACE_FORCE_GNU_ADDR2LINE_PATH)})) {
      return true;
    }
#  else
    if (cb(addr2line{base})) {
      return true;
    }
#  endif
  }

  if ((prog_name = getenv("LIBCXX_STACKTRACE_FORCE_APPLE_ATOS_PATH"))) {
    if (cb(atos{base, prog_name})) {
      return true;
    }
  } else {
#  if defined(LIBCXX_STACKTRACE_FORCE_APPLE_ATOS_PATH)
    if (cb(atos{base, STRINGIFY(LIBCXX_STACKTRACE_FORCE_APPLE_ATOS_PATH)})) {
      return true;
    }
#  else
    if (cb(atos{base})) {
      return true;
    }
#  endif
  }

  return false; // nothing succeeded
}

} // namespace

bool file_actions::initFileActions() {
  if (!fa_initialized_) {
    if (posix_spawn_file_actions_init(&fa_)) {
      return false;
    }
    fa_initialized_ = true;
  }
  return true;
}

file_actions::~file_actions() { posix_spawn_file_actions_destroy(&fa_); }

bool file_actions::addClose(int fd) { return initFileActions() && (posix_spawn_file_actions_addclose(&fa_, fd) == 0); }

bool file_actions::addDup2(int fd, int std_fd) {
  return initFileActions() && (posix_spawn_file_actions_adddup2(&fa_, fd, std_fd) == 0);
}

fd file_actions::redirectOutFD() {
  int fds[2];
  if (::pipe(fds)) {
    return {}; // return invalid FD
  }
  addClose(fds[0]);
  addDup2(fds[1], 1);
  return {fds[0]};
}

pspawn::~pspawn() {
  if (pid_) {
    kill(pid_, SIGTERM);
    wait();
  }
}

bool pspawn::spawn(base::list<base::str> const& argStrings) {
  base::vec<char const*> argv = tool_.base_.make_vec<char const*>();
  argv.reserve(argStrings.size() + 1);
  for (auto const& str : argStrings) {
    argv.push_back(str.data());
  }
  argv.push_back(nullptr);
  return posix_spawnp(&pid_, argv[0], &fa_.fa_, nullptr, const_cast<char**>(argv.data()), nullptr) == 0;
}

int pspawn::wait() {
  int status;
  waitpid(pid_, &status, 0);
  return status;
}

bool spawner::resolve_lines() {
  return try_tools(base_, [&](tool const& prog) {
    char buf[512];
    pspawn_tool proc(prog, base_, buf, sizeof(buf));
    return proc.run();
  });
}

base::list<base::str> llvm_symbolizer::buildArgs(base& base) const {
  auto ret = base_.make_list<base::str>();
  ret.push_back(base_.make_str(progName_));
  ret.push_back(base_.make_str("--demangle"));
  ret.push_back(base_.make_str("--no-inlines"));
  ret.push_back(base_.make_str("--verbose"));
  ret.push_back(base_.make_str("--relativenames"));
  ret.push_back(base_.make_str("--functions=short"));
  for (auto& st_entry : base.__entries_) {
    auto& entry      = (entry_base&)st_entry;
    auto addr_string = hex_string(base_, entry.__addr_unslid_);
    if (entry.__file_) {
      auto arg = base_.make_str();
      arg.reserve(entry.__file_->size() + 40);
      arg += "FILE:";
      arg += *entry.__file_;
      arg += " ";
      arg += addr_string;
      ret.push_back(arg);
    } else {
      ret.push_back(addr_string);
    }
  }
  return ret;
}

void llvm_symbolizer::parseOutput(base& base, __stacktrace::entry_base& entry, std::istream& output) const {
  // clang-format off
/*
With "--verbose", parsing is a little easier, or at least, more reliable;
probably the best solution (until we have a JSON parser).
Example output, verbatim, between the '---' lines:
---
test1<test_alloc<std::__1::stackbuilder_entry> >
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

  auto line = base.make_str();
  line.reserve(512);
  std::string_view tmp;
  while (true) {
    std::getline(output, line);
    while (!line.empty() && isspace(line.back())) {
      line.pop_back();
    }
    if (line.empty()) {
      return; // done
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
        entry.__file_ = base.make_str(tmp);
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

base::list<base::str> addr2line::buildArgs(base& base) const {
  auto ret = base.make_list<base::str>();
  if (base.__main_prog_path_.empty()) {
    // Should not have reached here but be graceful anyway
    ret.push_back(base.make_str("/bin/false"));
    return ret;
  }

  ret.push_back(base.make_str(progName_));
  ret.push_back(base.make_str("--functions"));
  ret.push_back(base.make_str("--demangle"));
  ret.push_back(base.make_str("--basenames"));
  ret.push_back(base.make_str("--pretty-print")); // This "human-readable form" is easier to parse
  ret.push_back(base.make_str("-e"));
  ret.push_back(base.__main_prog_path_);
  for (auto& entry : base.__entries_) {
    ret.push_back(hex_string(base, ((entry_base&)entry).__addr_unslid_));
  }
  return ret;
}

void addr2line::parseOutput(base& base, entry_base& entry, std::istream& output) const {
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

  auto line = base.make_str();
  line.reserve(512);
  std::getline(output, line);
  while (!line.empty() && isspace(line.back())) {
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
    entry.__desc_ = base.make_str(string_view(line).substr(0, sepIndex));
  }
  auto fileBegin = sepIndex + 4;
  if (fileBegin >= line.size()) {
    return;
  }
  auto fileline = base.make_str(string_view(line).substr(fileBegin));
  auto colon    = fileline.find_last_of(":");
  if (colon > 0 && !fileline.starts_with("?")) {
    entry.__file_ = base.make_str(string_view(fileline).substr(0, colon));
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

base::list<base::str> atos::buildArgs(base& base) const {
  auto ret = base.make_list<base::str>();
  ret.push_back(base.make_str(progName_));
  ret.push_back(base.make_str("-p"));
  ret.push_back(u64_string(base, getpid()));
  // TODO(stackcx23): Allow options in env, e.g. LIBCPP_STACKTRACE_OPTIONS=FullPath
  // ret.push_back("--fullPath");
  for (auto& entry : base.__entries_) {
    ret.push_back(hex_string(base, ((entry_base&)entry).__addr_actual_));
  }
  return ret;
}

void atos::parseOutput(base& base, entry_base& entry, std::istream& output) const {
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

  auto line = base.make_str();
  line.reserve(512);
  std::getline(output, line);
  while (!line.empty() && isspace(line.back())) {
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
  if (entry.__desc_->empty() && !sym.empty()) {
    entry.__desc_ = base.make_str(sym);
  }

  std::string_view file{fileBegin, size_t(lastColon - fileBegin)};
  if (file != "?" && file != "??" && !file.empty()) {
    entry.__file_ = base.make_str(file);
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
