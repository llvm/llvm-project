//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_CHECK_ASSERTION_H
#define TEST_SUPPORT_CHECK_ASSERTION_H

#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>

#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include "test_macros.h"
#include "test_allocator.h"

#ifndef _LIBCPP_VERSION
# error "This header may only be used for libc++ tests"
#endif

#if TEST_STD_VER < 11
# error "C++11 or greater is required to use this header"
#endif

struct AssertionInfoMatcher {
  static const int any_line = -1;
  static constexpr const char* any_file = "*";
  static constexpr const char* any_msg = "*";

  constexpr AssertionInfoMatcher() : is_empty_(true), msg_(any_msg, __builtin_strlen(any_msg)), file_(any_file, __builtin_strlen(any_file)), line_(any_line) { }
  constexpr AssertionInfoMatcher(const char* msg, const char* file = any_file, int line = any_line)
    : is_empty_(false), msg_(msg, __builtin_strlen(msg)), file_(file, __builtin_strlen(file)), line_(line) {}

  bool Matches(char const* file, int line, char const* message) const {
    assert(!empty() && "empty matcher");

    if (CheckLineMatches(line) && CheckFileMatches(file) && CheckMessageMatches(message))
        return true;
    // Write to stdout because that's the file descriptor captured by the parent
    // process.
    std::printf("Failed to match assertion info!\n%s\nVS\n%s:%d (%s)\n", ToString().data(), file, line, message);
    return false;
  }

  std::string ToString() const {
    std::string result = "msg = \""; result += msg_; result += "\"\n";
    result += "line = " + (line_ == any_line ? "'*'" : std::to_string(line_)) + "\n";
    result += "file = " + (file_ == any_file ? "'*'" : std::string(file_));
    return result;
  }

  bool empty() const { return is_empty_; }
private:
  bool CheckLineMatches(int got_line) const {
    if (line_ == any_line)
      return true;
    return got_line == line_;
  }

  bool CheckFileMatches(std::string_view got_file) const {
    assert(!empty() && "empty matcher");
    if (file_ == any_file)
      return true;
    std::size_t found_at = got_file.find(file_);
    if (found_at == std::string_view::npos)
      return false;
    // require the match start at the beginning of the file or immediately after
    // a directory separator.
    if (found_at != 0) {
      char last_char = got_file[found_at - 1];
      if (last_char != '/' && last_char != '\\')
        return false;
    }
    // require the match goes until the end of the string.
    return got_file.substr(found_at) == file_;
  }

  bool CheckMessageMatches(std::string_view got_msg) const {
    assert(!empty() && "empty matcher");
    if (msg_ == any_msg)
      return true;
    std::size_t found_at = got_msg.find(msg_);
    if (found_at == std::string_view::npos)
      return false;
    // Allow any match
    return true;
  }
private:
  bool is_empty_;
  std::string_view msg_;
  std::string_view file_;
  int line_;
};

static constexpr AssertionInfoMatcher AnyMatcher(AssertionInfoMatcher::any_msg);

inline AssertionInfoMatcher& GlobalMatcher() {
  static AssertionInfoMatcher GMatch;
  return GMatch;
}

struct DeathTest {
  enum ResultKind {
    RK_DidNotDie, RK_MatchFound, RK_MatchFailure, RK_SetupFailure, RK_Unknown
  };

  static const char* ResultKindToString(ResultKind RK) {
#define CASE(K) case K: return #K
    switch (RK) {
    CASE(RK_MatchFailure);
    CASE(RK_DidNotDie);
    CASE(RK_SetupFailure);
    CASE(RK_MatchFound);
    CASE(RK_Unknown);
    }
    return "not a result kind";
  }

  static bool IsValidResultKind(int val) {
    return val >= RK_DidNotDie && val <= RK_Unknown;
  }

  DeathTest(AssertionInfoMatcher const& Matcher) : matcher_(Matcher) {}

  template <class Func>
  ResultKind Run(Func&& f) {
    int pipe_res = pipe(stdout_pipe_fd_);
    assert(pipe_res != -1 && "failed to create pipe");
    pipe_res = pipe(stderr_pipe_fd_);
    assert(pipe_res != -1 && "failed to create pipe");
    pid_t child_pid = fork();
    assert(child_pid != -1 &&
        "failed to fork a process to perform a death test");
    child_pid_ = child_pid;
    if (child_pid_ == 0) {
      RunForChild(std::forward<Func>(f));
      assert(false && "unreachable");
    }
    return RunForParent();
  }

  int getChildExitCode() const { return exit_code_; }
  std::string const& getChildStdOut() const { return stdout_from_child_; }
  std::string const& getChildStdErr() const { return stderr_from_child_; }
private:
  template <class Func>
  TEST_NORETURN void RunForChild(Func&& f) {
    close(GetStdOutReadFD()); // don't need to read from the pipe in the child.
    close(GetStdErrReadFD());
    auto DupFD = [](int DestFD, int TargetFD) {
      int dup_result = dup2(DestFD, TargetFD);
      if (dup_result == -1)
        std::exit(RK_SetupFailure);
    };
    DupFD(GetStdOutWriteFD(), STDOUT_FILENO);
    DupFD(GetStdErrWriteFD(), STDERR_FILENO);

    GlobalMatcher() = matcher_;
    f();
    std::exit(RK_DidNotDie);
  }

  static std::string ReadChildIOUntilEnd(int FD) {
    std::string error_msg;
    char buffer[256];
    int num_read;
    do {
      while ((num_read = read(FD, buffer, 255)) > 0) {
        buffer[num_read] = '\0';
        error_msg += buffer;
      }
    } while (num_read == -1 && errno == EINTR);
    return error_msg;
  }

  void CaptureIOFromChild() {
    close(GetStdOutWriteFD()); // no need to write from the parent process
    close(GetStdErrWriteFD());
    stdout_from_child_ = ReadChildIOUntilEnd(GetStdOutReadFD());
    stderr_from_child_ = ReadChildIOUntilEnd(GetStdErrReadFD());
    close(GetStdOutReadFD());
    close(GetStdErrReadFD());
  }

  ResultKind RunForParent() {
    CaptureIOFromChild();

    int status_value;
    pid_t result = waitpid(child_pid_, &status_value, 0);
    assert(result != -1 && "there is no child process to wait for");

    if (WIFEXITED(status_value)) {
      exit_code_ = WEXITSTATUS(status_value);
      if (!IsValidResultKind(exit_code_))
        return RK_Unknown;
      return static_cast<ResultKind>(exit_code_);
    }
    return RK_Unknown;
  }

  DeathTest(DeathTest const&) = delete;
  DeathTest& operator=(DeathTest const&) = delete;

  int GetStdOutReadFD() const {
    return stdout_pipe_fd_[0];
  }

  int GetStdOutWriteFD() const {
    return stdout_pipe_fd_[1];
  }

  int GetStdErrReadFD() const {
    return stderr_pipe_fd_[0];
  }

  int GetStdErrWriteFD() const {
    return stderr_pipe_fd_[1];
  }
private:
  AssertionInfoMatcher matcher_;
  pid_t child_pid_ = -1;
  int exit_code_ = -1;
  int stdout_pipe_fd_[2];
  int stderr_pipe_fd_[2];
  std::string stdout_from_child_;
  std::string stderr_from_child_;
};

void std::__libcpp_verbose_abort(char const* format, ...) {
  assert(!GlobalMatcher().empty());

  // Extract information from the error message. This has to stay synchronized with
  // how we format assertions in the library.
  va_list list;
  va_start(list, format);
  char const* file = va_arg(list, char const*);
  int line = va_arg(list, int);
  char const* expression = va_arg(list, char const*); (void)expression;
  char const* message = va_arg(list, char const*);
  va_end(list);

  if (GlobalMatcher().Matches(file, line, message)) {
    std::exit(DeathTest::RK_MatchFound);
  }
  std::exit(DeathTest::RK_MatchFailure);
}

template <class Func>
inline bool ExpectDeath(const char* stmt, Func&& func, AssertionInfoMatcher Matcher) {
  DeathTest DT(Matcher);
  DeathTest::ResultKind RK = DT.Run(func);
  auto OnFailure = [&](const char* msg) {
    std::fprintf(stderr, "EXPECT_DEATH( %s ) failed! (%s)\n\n", stmt, msg);
    if (RK != DeathTest::RK_Unknown) {
      std::fprintf(stderr, "child exit code: %d\n", DT.getChildExitCode());
    }
    if (!DT.getChildStdErr().empty()) {
      std::fprintf(stderr, "---------- standard err ----------\n%s\n", DT.getChildStdErr().c_str());
    }
    if (!DT.getChildStdOut().empty()) {
      std::fprintf(stderr, "---------- standard out ----------\n%s\n", DT.getChildStdOut().c_str());
    }
    return false;
  };
  switch (RK) {
  case DeathTest::RK_MatchFound:
    return true;
  case DeathTest::RK_SetupFailure:
    return OnFailure("child failed to setup test environment");
  case DeathTest::RK_Unknown:
      return OnFailure("reason unknown");
  case DeathTest::RK_DidNotDie:
      return OnFailure("child did not die");
  case DeathTest::RK_MatchFailure:
      return OnFailure("matcher failed");
  }
  assert(false && "unreachable");
}

template <class Func>
inline bool ExpectDeath(const char* stmt, Func&& func) {
  return ExpectDeath(stmt, func, AnyMatcher);
}

/// Assert that the specified expression throws a libc++ debug exception.
#define EXPECT_DEATH(...) assert((ExpectDeath(#__VA_ARGS__, [&]() { __VA_ARGS__; } )))

#define EXPECT_DEATH_MATCHES(Matcher, ...) assert((ExpectDeath(#__VA_ARGS__, [&]() { __VA_ARGS__; }, Matcher)))

#define TEST_LIBCPP_ASSERT_FAILURE(expr, message) assert((ExpectDeath(#expr, [&]() { (void)(expr); }, AssertionInfoMatcher(message))))

#endif // TEST_SUPPORT_CHECK_ASSERTION_H
