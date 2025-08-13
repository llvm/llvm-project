//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_TOOLS_TOOL_DEFN
#define _LIBCPP_STACKTRACE_TOOLS_TOOL_DEFN

#include <__config>
#include <memory>

#if __has_include(<spawn.h>) && _LIBCPP_STACKTRACE_ALLOW_TOOLS_AT_RUNTIME

#  include <__stacktrace/basic_stacktrace.h>
#  include <__stacktrace/stacktrace_entry.h>
#  include <cctype>
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

#  include "stacktrace/fd.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct tool_base {
  constexpr static size_t k_max_argv_ = base::__absolute_max_depth + 10;
  base& base_;
  arena& arena_;
  char const* tool_prog_;
  str argvs_[k_max_argv_]{};         // will hold our generated arg strings
  char* argv_[k_max_argv_]{nullptr}; // refers to argvs_ strings as char** (includes null terminator)
  size_t argc_{0};                   // number of args.  Note: argv_[argc_] is nullptr

  _LIBCPP_HIDE_FROM_ABI tool_base(base& base, arena& arena, char const* tool_prog)
      : base_(base), arena_(arena), tool_prog_(tool_prog) {
    argv_[0] = nullptr;
  }

  _LIBCPP_HIDE_FROM_ABI void push_arg(std::string_view sv) {
    _LIBCPP_ASSERT(argc_ < k_max_argv_ - 1, "too many args");
    argvs_[argc_]  = sv;                   // Have to copy the string_view into a new string
    argv_[argc_]   = argvs_[argc_].data(); // then we have a char pointer into that string
    argv_[++argc_] = nullptr;              // ensure there's always trailing null after last arg
  }

  _LIBCPP_HIDE_FROM_ABI void push_arg(str __str) { push_arg(std::string_view{__str.data(), __str.size()}); }

  template <typename... _Args>
  _LIBCPP_HIDE_FROM_ABI void push_arg(char const* format, _Args&&... args) {
    push_arg(str::makef(format, std::forward<_Args>(args)...));
  }

  // Helper functions for dealing with string views.
  // All these take a string_view (by copy) and return a modified view.  Inputs are validated
  // and if invalid, the function returns an empty view (instead of throwing).

  /** Drop `n` chars from the start of the string; empty string if `n` exceeds string size */
  _LIBCPP_HIDE_FROM_ABI static string_view ldrop(string_view sv, size_t n = 1) {
    sv.remove_prefix(std::min(sv.size(), n));
    return sv;
  }

  /** Drop `n` chars from the end of the string; empty string if `n` exceeds string size */
  _LIBCPP_HIDE_FROM_ABI static string_view rdrop(string_view sv, size_t n = 1) {
    sv.remove_suffix(std::min(sv.size(), n));
    return sv;
  }

  /** Strip whitespace from the start of the string */
  _LIBCPP_HIDE_FROM_ABI static string_view lstrip(string_view sv) {
    while (!sv.empty() && isspace(sv.front())) {
      sv = ldrop(sv);
    };
    return sv;
  }

  /** Strip whitespace from the back of the string */
  _LIBCPP_HIDE_FROM_ABI static string_view rstrip(string_view sv) {
    while (!sv.empty() && isspace(sv.back())) {
      sv = rdrop(sv);
    };
    return sv;
  }

  /** Strip whitespace from the start and end of the string */
  _LIBCPP_HIDE_FROM_ABI static string_view strip(string_view sv) { return lstrip(rstrip(sv)); }

  /** Drop prefix if exists; if not found, and if required, return empty (failure); else original arg */
  _LIBCPP_HIDE_FROM_ABI static string_view drop_prefix(string_view sv, string_view pre, bool required = true) {
    if (sv.starts_with(pre)) {
      return ldrop(sv, pre.size());
    }
    return required ? string_view{} : sv;
  }

  /** Drop suffix if exists; if not found, and if required, return empty (failure); else original arg */
  _LIBCPP_HIDE_FROM_ABI static string_view drop_suffix(string_view sv, string_view suf, bool required = true) {
    if (sv.ends_with(suf)) {
      return rdrop(sv, suf.size());
    }
    return required ? string_view{} : sv;
  }
};

/** Set up a `posix_spawn_file_actions_t` for use with a symbolizer with redirected stdout. */
struct file_actions {
  optional<posix_spawn_file_actions_t> fa_{};
  fd stdout_read_;  // read end of subprocess's stdout, IFF redir_stdout used
  fd stdout_write_; // write end of subprocess's stdout, IFF redir_stdout used
  int errno_{};     // set to nonzero if any of these C calls failed

  _LIBCPP_HIDE_FROM_ABI bool failed() const { return errno_; }

  _LIBCPP_HIDE_FROM_ABI posix_spawn_file_actions_t* fa() {
    if (!fa_) {
      fa_.emplace();
      if (posix_spawn_file_actions_init(&fa_.value())) {
        errno_ = errno;
        _LIBCPP_ASSERT(false, "file_actions_init failed");
        fa_.reset();
      }
    }
    return fa_ ? &fa_.value() : nullptr;
  }

  _LIBCPP_HIDE_FROM_ABI ~file_actions() {
    if (fa_) {
      // Do best-effort teardown, ignore errors
      (void)posix_spawn_file_actions_destroy(&fa_.value());
      fa_.reset();
    }
  }

  _LIBCPP_HIDE_FROM_ABI file_actions()                               = default;
  _LIBCPP_HIDE_FROM_ABI file_actions(file_actions const&)            = delete;
  _LIBCPP_HIDE_FROM_ABI file_actions& operator=(file_actions const&) = delete;

  _LIBCPP_HIDE_FROM_ABI file_actions(file_actions&& rhs) {
    fa_ = std::move(rhs.fa_);
    rhs.fa_.reset();
    stdout_read_  = std::move(rhs.stdout_read_);
    stdout_write_ = std::move(rhs.stdout_write_);
    errno_        = rhs.errno_;
  }

  _LIBCPP_HIDE_FROM_ABI file_actions& operator=(file_actions&& rhs) {
    return (std::addressof(rhs) == this) ? *this : *(new (this) file_actions(std::move(rhs)));
  }

  // These have no effect if this is already in `failed` state.

  _LIBCPP_HIDE_FROM_ABI file_actions& no_stdin() {
    if (!failed() && posix_spawn_file_actions_adddup2(fa(), fd::null_fd(), STDIN_FILENO)) {
      _LIBCPP_ASSERT(false, "no_stdin: adddup2 failed");
      errno_ = errno;
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI file_actions& no_stdout() {
    if (!failed() && posix_spawn_file_actions_adddup2(fa(), fd::null_fd(), STDOUT_FILENO)) {
      _LIBCPP_ASSERT(false, "no_stdout: adddup2 failed");
      errno_ = errno;
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI file_actions& no_stderr() {
    if (!failed() && posix_spawn_file_actions_adddup2(fa(), fd::null_fd(), STDERR_FILENO)) {
      _LIBCPP_ASSERT(false, "no_stderr: adddup2 failed");
      errno_ = errno;
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI file_actions& redir_stdout() {
    if (!failed() && fd::pipe_pair(stdout_read_, stdout_write_)) {
      _LIBCPP_ASSERT(false, "redir_stdout: pipe failed");
      errno_ = errno;
    } else if (!failed() && posix_spawn_file_actions_adddup2(fa(), stdout_write_, STDOUT_FILENO)) {
      _LIBCPP_ASSERT(false, "redir_stdout: adddup2 failed");
      errno_ = errno;
    } else if (!failed() && posix_spawn_file_actions_addclose(fa(), stdout_read_)) {
      _LIBCPP_ASSERT(false, "redir_stdout: pipe failed");
      errno_ = errno;
    }
    return *this;
  }
};

/** While in-scope, this enables SIGCHLD default handling (allowing `waitpid` to work).
Restores the old signal action on destruction.

XXX Thread safety issue almost certainly exists here
*/
struct sigchld_enable {
  struct sigaction old_;

  _LIBCPP_HIDE_FROM_ABI ~sigchld_enable() {
    int res = sigaction(SIGCHLD, &old_, nullptr); // restore old behavior
    _LIBCPP_ASSERT(!res, "~sigchld_enable: sigaction failed");
  }

  _LIBCPP_HIDE_FROM_ABI sigchld_enable() {
    struct sigaction act;
    sigemptyset(&act.sa_mask);
    act.sa_flags   = 0;
    act.sa_handler = SIG_DFL;
    int res        = sigaction(SIGCHLD, &act, &old_);
    _LIBCPP_ASSERT(!res, "sigchld_enable: sigaction failed");
  }
};

struct pid_waiter {
  pid_t pid_{};
  int status_{}; // value is valid iff wait() completed
  int errno_{};  // set to nonzero if any of these C calls failed
  bool done_{};

  _LIBCPP_HIDE_FROM_ABI operator pid_t() const { return pid_; }
  _LIBCPP_HIDE_FROM_ABI bool running() const { return pid_ && !kill(pid_, 0); }
  _LIBCPP_HIDE_FROM_ABI bool failed() const { return errno_; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI int wait() {
    while (!done_) { // Until successful waitpid, or a hard error:
      int result;
      if (waitpid(pid_, &result, 0) == pid_) { // attempt a blocking wait, updates status_
        if (WIFEXITED(result)) {               // process exited? (not signaled)
          status_ = WEXITSTATUS(result);       // get exit code
          done_   = true;                      //
        } else if (WIFSIGNALED(result)) {      // if signaled:
          status_ = -WTERMSIG(result);         // use negative to indicate signal
          done_   = true;                      //
        }
      } else if (errno != EINTR) { // for errors other than interrupted syscall (which we retry),
        errno_  = errno;           // record the error, putting this in `failed` state
        done_   = true;            // don't bother attempting another wait
        status_ = -1;              // nonzero bogus value
      }
    }
    return status_;
  }

  _LIBCPP_HIDE_FROM_ABI ~pid_waiter() {
    if (pid_ && !done_) {
      // this represents a valid but non-waited pid
      if (running()) {
        kill(pid_, SIGKILL);
      }
      (void)/* ignore status */ wait();
    }
  }
};

struct spawner {
  tool_base& tool_;
  base& base_;
  file_actions fa_{};          // redirects stdout for us
  char cbuf_[4 << 10];         // buffer space for the streambuf:
  fd::streambuf sbuf_;         // streambuf interface for the istream:
  fd::istream stream_;         // istream interface from which we can `getline`
  sigchld_enable chld_enable_; // temporarily enables SIGCHLD so `waitpid` works
  pid_waiter pid_{0};          // set during successful `spawn`, can `waitpid` automatically
  int errno_{};                // set to nonzero if any of these C calls failed

  _LIBCPP_HIDE_FROM_ABI bool failed() const { return errno_; }

  _LIBCPP_HIDE_FROM_ABI spawner(tool_base& tool, base& base)
      : tool_{tool},
        base_(base),
        fa_(std::move(file_actions().no_stdin().no_stderr().redir_stdout())),
        sbuf_(fa_.stdout_read_, cbuf_, sizeof(cbuf_)),
        stream_(sbuf_) {
    // Inherit any errors from during fileactions setup
    errno_ = fa_.errno_;
    if (!failed() && posix_spawnp(&pid_.pid_, tool_.argv_[0], fa_.fa(), nullptr, tool_.argv_, nullptr)) {
      _LIBCPP_ASSERT(false, "spawner: posix_spawnp failed");
      errno_ = errno;
    } else if (!failed() && close(fa_.stdout_write_)) {
      _LIBCPP_ASSERT(false, "spawner: close failed");
      errno_ = errno;
    }
  }
};

template <class T>
struct __executable_name {
  _LIBCPP_EXPORTED_FROM_ABI static char const* get() {
    auto* env_var = T::__override_prog_env;
    if (env_var) {
      auto* env_val = getenv(env_var);
      if (env_val) {
        return env_val;
      }
    }
    return T::__default_prog_name;
  }
};

/** Run `/usr/bin/env $TOOL --help`.  Succeeds iff `env` can find the tool in path, and tool exits without error. */
inline bool _LIBCPP_EXPORTED_FROM_ABI __executable_works(char const* prog_name) {
  char const* argv[4] = {"/usr/bin/env", prog_name, "--help", nullptr};
  pid_waiter pid;
  auto fa = std::move(file_actions().no_stdin().no_stdout().no_stderr());
  return posix_spawn(&pid.pid_, argv[0], fa.fa(), nullptr, const_cast<char**>(argv), nullptr)
           ? false        // spawn failed (don't care why), can't run prog
           : !pid.wait(); // otherwise, tool should return without error
}

/** Checks (and memoizes) whether tool's binary exists and runs */
template <class T>
inline bool _LIBCPP_EXPORTED_FROM_ABI __has_working_executable() {
  static bool ret = __executable_works(__executable_name<T>::get());
  return ret;
}

template <class T>
bool _LIBCPP_EXPORTED_FROM_ABI __run_tool(base&, arena&);

struct llvm_symbolizer;
extern template struct __executable_name<llvm_symbolizer>;
extern template bool _LIBCPP_EXPORTED_FROM_ABI __has_working_executable<llvm_symbolizer>();
template <>
bool _LIBCPP_EXPORTED_FROM_ABI __run_tool<llvm_symbolizer>(base&, arena&);

struct addr2line;
extern template struct __executable_name<addr2line>;
extern template bool _LIBCPP_EXPORTED_FROM_ABI __has_working_executable<addr2line>();
template <>
bool _LIBCPP_EXPORTED_FROM_ABI __run_tool<addr2line>(base&, arena&);

struct atos;
extern template struct __executable_name<atos>;
extern template bool _LIBCPP_EXPORTED_FROM_ABI __has_working_executable<atos>();
template <>
bool _LIBCPP_EXPORTED_FROM_ABI __run_tool<atos>(base&, arena&);

struct llvm_symbolizer : tool_base {
  constexpr static char const* __default_prog_name = "llvm-symbolizer";
  constexpr static char const* __override_prog_env = "LIBCXX_STACKTRACE_FORCE_LLVM_SYMBOLIZER_PATH";

  _LIBCPP_HIDE_FROM_ABI llvm_symbolizer(base& base, arena& arena)
      : tool_base{base, arena, __executable_name<llvm_symbolizer>::get()} {}
  _LIBCPP_HIDE_FROM_ABI bool build_argv();
  _LIBCPP_HIDE_FROM_ABI void parse(entry_base** entry_iter, std::string_view view) const;
};

struct addr2line : tool_base {
  constexpr static char const* __default_prog_name = "addr2line";
  constexpr static char const* __override_prog_env = "LIBCXX_STACKTRACE_FORCE_GNU_ADDR2LINE_PATH";

  _LIBCPP_HIDE_FROM_ABI addr2line(base& base, arena& arena)
      : tool_base{base, arena, __executable_name<addr2line>::get()} {}
  _LIBCPP_HIDE_FROM_ABI bool build_argv();
  _LIBCPP_HIDE_FROM_ABI void parse_sym(entry_base& entry, std::string_view view) const;
  _LIBCPP_HIDE_FROM_ABI void parse_loc(entry_base& entry, std::string_view view) const;
};

struct atos : tool_base {
  constexpr static char const* __default_prog_name = "atos";
  constexpr static char const* __override_prog_env = "LIBCXX_STACKTRACE_FORCE_APPLE_ATOS_PATH";

  _LIBCPP_HIDE_FROM_ABI atos(base& base, arena& arena) : tool_base{base, arena, __executable_name<atos>::get()} {}
  _LIBCPP_HIDE_FROM_ABI bool build_argv();
  _LIBCPP_HIDE_FROM_ABI void parse(entry_base& entry, std::string_view view) const;
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // __has_include(<spawn.h>) && _LIBCPP_STACKTRACE_ALLOW_TOOLS_AT_RUNTIME

#endif // _LIBCPP_STACKTRACE_TOOLS_TOOL_DEFN
