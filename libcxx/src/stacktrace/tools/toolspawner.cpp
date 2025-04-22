//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/config.h"

#if defined(_LIBCPP_STACKTRACE_CAN_SPAWN_TOOLS)

#  include <__config>
#  include <__config_site>
#  include <cassert>
#  include <cerrno>
#  include <csignal>
#  include <cstddef>
#  include <cstdlib>
#  include <spawn.h>
#  include <string>
#  include <sys/fcntl.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>

#  include "../common/failed.h"
#  include "pspawn.h"

#  include <__stacktrace/entry.h>

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

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
