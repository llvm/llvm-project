//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_PSPAWN_H
#define _LIBCPP_STACKTRACE_PSPAWN_H

#include "../common/config.h"

#if defined(_LIBCPP_STACKTRACE_CAN_SPAWN_TOOLS)

#  include <__config>
#  include <__config_site>
#  include <cassert>
#  include <cerrno>
#  include <csignal>
#  include <cstddef>
#  include <cstdlib>
#  include <list>
#  include <spawn.h>
#  include <string>
#  include <sys/fcntl.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  include <vector>

#  include <__stacktrace/context.h>
#  include <__stacktrace/entry.h>

#  include "../common/debug.h"
#  include "../common/failed.h"
#  include "../common/fd.h"
#  include "tools.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

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

  void spawn(std::pmr::list<std::pmr::string> const& argStrings) {
    std::pmr::vector<char const*> argv{argStrings.get_allocator()};
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
  context& cx_;
  fd fd_;
  fd_streambuf buf_;
  fd_istream stream_;

  pspawn_tool(tool const& a2l, context& cx, char* buf, size_t size)
      : pspawn{a2l}, cx_(cx), fd_(fa_.redirectOutFD()), buf_(fd_, buf, size), stream_(buf_) {
    if (!debug::enabled()) {
      fa_.redirectErrNull();
    }
    fa_.redirectInNull();
  }

  void run() {
    // Cannot run "addr2line" or similar without addresses, since we
    // provide them in argv, and if there are none passed in argv, the
    // tool will try to read from stdin and hang.
    if (cx_.__entries_.empty()) {
      return;
    }

    auto argStrings = tool_.buildArgs(cx_);
    if (debug::enabled()) {
      debug() << "Trying to get stacktrace using:";
      for (auto& str : argStrings) {
        debug() << " \"" << str << '"';
      }
      debug() << '\n';
    }

    spawn(argStrings);

    for (auto& entry : cx_.__entries_) {
      tool_.parseOutput(cx_, entry, stream_);
    }
  }
};

struct spawner {
  context& cx_;
  void resolve_lines();
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif

#endif // _LIBCPP_STACKTRACE_PSPAWN_H
