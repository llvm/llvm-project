//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_TOOLS_H
#define _LIBCPP_STACKTRACE_TOOLS_H

#include <__config>
#include <__config_site>
#include <cassert>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <spawn.h>
#include <string>
#include <sys/fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <__stacktrace/base.h>
#include <__stacktrace/basic.h>
#include <__stacktrace/entry.h>

#include "stacktrace/utils/debug.h"
#include "stacktrace/utils/failed.h"
#include "stacktrace/utils/fd.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct tool {
  alloc& alloc_;
  char const* progName_;

  tool(alloc& alloc, char const* progName) : alloc_(alloc), progName_(progName) {}
  virtual ~tool() = default;

  /** Construct complete `argv` for the spawned process.
  Includes the program name at argv[0], followed by flags */
  virtual alloc::list<alloc::str> buildArgs(builder& trace) const = 0;

  /** Parse line(s) output by the tool, and modify `entry`. */
  virtual void parseOutput(builder& trace, entry_base& entry, std::istream& output) const = 0;
};

struct llvm_symbolizer : tool {
  virtual ~llvm_symbolizer() = default;
  explicit llvm_symbolizer(alloc& alloc) : llvm_symbolizer(alloc, "llvm_symbolizer") {}
  llvm_symbolizer(alloc& alloc, char const* progName) : tool{alloc, progName} {}
  alloc::list<alloc::str> buildArgs(builder& trace) const override;
  void parseOutput(builder& trace, entry_base& entry, std::istream& output) const override;
};

struct addr2line : tool {
  virtual ~addr2line() = default;
  explicit addr2line(alloc& alloc) : addr2line(alloc, "addr2line") {}
  addr2line(alloc& alloc, char const* progName) : tool{alloc, progName} {}
  alloc::list<alloc::str> buildArgs(builder& trace) const override;
  void parseOutput(builder& trace, entry_base& entry, std::istream& stream) const override;
};

struct atos : tool {
  virtual ~atos() = default;
  explicit atos(alloc& alloc) : atos(alloc, "atos") {}
  atos(alloc& alloc, char const* progName) : tool{alloc, progName} {}
  alloc::list<alloc::str> buildArgs(builder& trace) const override;
  void parseOutput(builder& trace, entry_base& entry, std::istream& output) const override;
};

struct file_actions {
  posix_spawn_file_actions_t fa_;

  file_actions() {
    if (posix_spawn_file_actions_init(&fa_)) {
      throw failed("posix_spawn_file_actions_init", errno);
    }
  }

  ~file_actions() { posix_spawn_file_actions_destroy(&fa_); }

  void addClose(int fd) {
    if (posix_spawn_file_actions_addclose(&fa_, fd)) {
      throw failed("posix_spawn_file_actions_addclose", errno);
    }
  }
  void addDup2(int fd, int stdfd) {
    if (posix_spawn_file_actions_adddup2(&fa_, fd, stdfd)) {
      throw failed("posix_spawn_file_actions_adddup2", errno);
    }
  }

  fd redirectOutFD() {
    int fds[2];
    if (::pipe(fds)) {
      throw failed("pipe", errno);
    }
    addClose(fds[0]);
    addDup2(fds[1], 1);
    return {fds[0]};
  }

  void redirectInNull() { addDup2(fd::null_fd(), 0); }
  void redirectOutNull() { addDup2(fd::null_fd(), 1); }
  void redirectErrNull() { addDup2(fd::null_fd(), 2); }
};

struct pspawn {
  tool const& tool_;
  pid_t pid_{0};
  file_actions fa_{};

  // TODO(stacktrace23): ignore SIGCHLD for spawned subprocess

  ~pspawn() {
    if (pid_) {
      kill(pid_, SIGTERM);
      wait();
    }
  }

  void spawn(alloc::list<alloc::str> const& argStrings) {
    alloc::vec<char const*> argv = tool_.alloc_.make_vec<char const*>();
    argv.reserve(argStrings.size() + 1);
    for (auto const& str : argStrings) {
      argv.push_back(str.data());
    }
    argv.push_back(nullptr);
    int err;
    if ((err = posix_spawnp(&pid_, argv[0], &fa_.fa_, nullptr, const_cast<char**>(argv.data()), nullptr))) {
      throw failed("posix_spawnp", err);
    }
  }

  int wait() {
    int status;
    waitpid(pid_, &status, 0);
    return status;
  }
};

struct pspawn_tool : pspawn {
  builder& builder_;
  fd fd_;
  fd_streambuf buf_;
  fd_istream stream_;

  pspawn_tool(tool const& a2l, builder& trace, char* buf, size_t size)
      : pspawn{a2l}, builder_(trace), fd_(fa_.redirectOutFD()), buf_(fd_, buf, size), stream_(buf_) {
    if (!debug::enabled()) {
      fa_.redirectErrNull();
    }
    fa_.redirectInNull();
  }

  void run() {
    // Cannot run "addr2line" or similar without addresses, since we
    // provide them in argv, and if there are none passed in argv, the
    // tool will try to read from stdin and hang.
    if (builder_.__entries_.empty()) {
      return;
    }

    auto argStrings = tool_.buildArgs(builder_);
    if (debug::enabled()) {
      debug() << "Trying to get stacktrace using:";
      for (auto& str : argStrings) {
        debug() << " \"" << str << '"';
      }
      debug() << '\n';
    }

    spawn(argStrings);

    auto end = builder_.__entries_.end();
    auto it  = builder_.__entries_.begin();
    while (it != end) {
      auto& entry = (entry_base&)(*it++);
      tool_.parseOutput(builder_, entry, stream_);
    }
  }
};

struct spawner {
  builder& builder_;
  void resolve_lines();
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_TOOLS_H
