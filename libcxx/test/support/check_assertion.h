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
#include <exception>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include "test_macros.h"
#include "test_allocator.h"

#if TEST_STD_VER < 11
# error "C++11 or greater is required to use this header"
#endif

struct AssertionInfoMatcher {
  // When printing the assertion message to `stderr`, delimit it with a marker to make it easier to match the message
  // later.
  static constexpr const char* Marker = "###";

  static const int any_line = -1;
  static constexpr const char* any_file = "*";
  static constexpr const char* any_msg = "*";

  constexpr AssertionInfoMatcher()
      : msg_(any_msg, __builtin_strlen(any_msg)),
        file_(any_file, __builtin_strlen(any_file)),
        line_(any_line) {}
  constexpr AssertionInfoMatcher(const char* msg, const char* file = any_file, int line = any_line)
      : is_empty_(false), msg_(msg, __builtin_strlen(msg)), file_(file, __builtin_strlen(file)), line_(line) {}

  bool CheckMatchInOutput(const std::string& output, std::string& error) const {
    // Extract information from the error message. This has to stay synchronized with how we format assertions in the
    // library.
    std::regex message_format(".*###\\n(.*):(\\d+): assertion (.*) failed: (.*)\\n###");

    std::smatch match_result;
    bool has_match = std::regex_match(output, match_result, message_format);
    assert(has_match);
    assert(match_result.size() == 5);

    const std::string& file = match_result[1];
    int line         = std::stoi(match_result[2]);
    // Omitting `expression` in `match_result[3]`
    const std::string& failure_reason = match_result[4];

    bool result = Matches(file, line, failure_reason);
    if (!result) {
      error = FormatMatchingError(file, line, failure_reason);
    }
    return result;
  }

  bool Matches(const std::string& file, int line, const std::string& message) const {
    assert(!empty() && "Empty matcher");
    return CheckLineMatches(line) && CheckFileMatches(file) && CheckMessageMatches(message);
  }

  std::string FormatMatchingError(const std::string& file, int line, const std::string& message) const {
    std::stringstream output;
    output //
        << "Expected message:   '" << msg_.data() << "'\n" //
        << "Actual message:     '" << message.c_str() << "'\n" //
        << "Expected location:   " << FormatLocation(file_, line_) << "\n" //
        << "Actual location:     " << FormatLocation(file, line) << "\n";
    return output.str();
  }

  static std::string FormatLocation(std::string_view file, int line) {
    std::string result;
    result += (file == any_file ? "*" : std::string(file)) + ":";
    result += (line == any_line ? "*" : std::to_string(line));
    return result;
  }

  bool empty() const { return is_empty_; }
  bool IsAnyMatcher() const {
    return msg_ == any_msg && file_ == any_file && line_ == any_line;
  }

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
  bool is_empty_ = true;;
  std::string_view msg_;
  std::string_view file_;
  int line_;
};

static constexpr AssertionInfoMatcher AnyMatcher(AssertionInfoMatcher::any_msg);

enum class DeathCause {
  // Valid causes
  VerboseAbort = 1,
  StdTerminate,
  Trap,
  // Errors
  DidNotDie,
  SetupFailure,
  Unknown
};

bool IsValidCause(DeathCause cause) {
  switch (cause) {
    case DeathCause::VerboseAbort:
    case DeathCause::StdTerminate:
    case DeathCause::Trap:
      return true;
    default:
      return false;
  }
}

std::string ToString(DeathCause cause) {
  switch (cause) {
    case DeathCause::VerboseAbort:
      return "verbose abort";
    case DeathCause::StdTerminate:
      return "`std::terminate`";
    case DeathCause::Trap:
      return "trap";
    default:
      return "<invalid cause>";
  }
}

TEST_NORETURN void StopChildProcess(DeathCause cause) {
  std::exit(static_cast<int>(cause));
}

DeathCause ConvertToDeathCause(int val) {
  if (val < static_cast<int>(DeathCause::VerboseAbort) || val > static_cast<int>(DeathCause::Unknown)) {
    return DeathCause::Unknown;
  }
  return static_cast<DeathCause>(val);
}

struct DeathTest {
  DeathTest() = default;
  DeathTest(DeathTest const&) = delete;
  DeathTest& operator=(DeathTest const&) = delete;

  template <class Func>
  DeathCause Run(Func&& f) {
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
        StopChildProcess(DeathCause::SetupFailure);
    };
    DupFD(GetStdOutWriteFD(), STDOUT_FILENO);
    DupFD(GetStdErrWriteFD(), STDERR_FILENO);

    f();
    StopChildProcess(DeathCause::DidNotDie);
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

  DeathCause RunForParent() {
    CaptureIOFromChild();

    int status_value;
    pid_t result = waitpid(child_pid_, &status_value, 0);
    assert(result != -1 && "there is no child process to wait for");

    if (WIFEXITED(status_value)) {
      exit_code_ = WEXITSTATUS(status_value);
      return ConvertToDeathCause(exit_code_);
    }

    if (WIFSIGNALED(status_value)) {
      exit_code_ = WTERMSIG(status_value);
      if (exit_code_ == SIGILL || exit_code_ == SIGTRAP) {
        return DeathCause::Trap;
      }
    }

    return DeathCause::Unknown;
  }

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
  pid_t child_pid_ = -1;
  int exit_code_ = -1;
  int stdout_pipe_fd_[2];
  int stderr_pipe_fd_[2];
  std::string stdout_from_child_;
  std::string stderr_from_child_;
};

#ifdef _LIBCPP_VERSION
void std::__libcpp_verbose_abort(char const* format, ...) {
  va_list args;
  va_start(args, format);

  std::fprintf(stderr, "%s\n", AssertionInfoMatcher::Marker);
  std::vfprintf(stderr, format, args);
  std::fprintf(stderr, "%s", AssertionInfoMatcher::Marker);

  va_end(args);

  StopChildProcess(DeathCause::VerboseAbort);
}
#endif // _LIBCPP_VERSION

[[noreturn]] inline void terminate_handler() {
  StopChildProcess(DeathCause::StdTerminate);
}

template <class Func>
inline bool ExpectDeath(DeathCause expected_cause, const char* stmt, Func&& func, AssertionInfoMatcher matcher) {
  assert(IsValidCause(expected_cause));

  std::set_terminate(terminate_handler);

  DeathTest test_case;
  DeathCause cause = test_case.Run(func);

  auto OnFailure = [&](std::string_view msg) {
    std::fprintf(stderr, "Failure: EXPECT_DEATH( %s ) failed!\n(cause: %s)\n\n", stmt, msg.data());

    if (cause != DeathCause::Unknown) {
      std::fprintf(stderr, "child exit code: %d\n", test_case.getChildExitCode());
    }
    std::fprintf(stderr, "---------- standard err ----------\n%s", test_case.getChildStdErr().c_str());
    std::fprintf(stderr, "\n----------------------------------\n");
    std::fprintf(stderr, "---------- standard out ----------\n%s", test_case.getChildStdOut().c_str());
    std::fprintf(stderr, "\n----------------------------------\n");

    return false;
  };

  auto CauseMismatchMessage = [](DeathCause expected, DeathCause actual) {
    std::stringstream output;
    output                                                    //
        << "Child died, but with a different death cause\n"   //
        << "Expected cause:   " << ToString(expected) << "\n" //
        << "Actual cause:     " << ToString(actual) << "\n";
    return output.str();
  };

  switch (cause) {
    case DeathCause::StdTerminate:
    case DeathCause::Trap:
      if (expected_cause != cause) {
        return OnFailure(CauseMismatchMessage(expected_cause, cause));
      }
      return true;

    case DeathCause::VerboseAbort: {
      if (expected_cause != cause) {
        return OnFailure(CauseMismatchMessage(expected_cause, cause));
      }
      std::string maybe_error;
      if (matcher.CheckMatchInOutput(test_case.getChildStdErr(), maybe_error)) {
        return true;
      }
      maybe_error = std::string("Child died, but with a different verbose abort message\n") + maybe_error;
      return OnFailure(maybe_error);
    }

    // Unexpected causes.
    case DeathCause::SetupFailure:
      return OnFailure("Child failed to set up test environment");
    case DeathCause::Unknown:
      return OnFailure("Cause unknown");
    case DeathCause::DidNotDie:
      return OnFailure("Child did not die");
  }

  assert(false && "unreachable");
}

template <class Func>
inline bool ExpectDeath(DeathCause expected_cause, const char* stmt, Func&& func) {
  return ExpectDeath(expected_cause, stmt, func, AnyMatcher);
}

// clang-format off

/// Assert that the specified expression aborts with the expected cause and, optionally, error message.
#define EXPECT_DEATH(...)                         \
    assert(( ExpectDeath(DeathCause::VerboseAbort, #__VA_ARGS__, [&]() { __VA_ARGS__; } ) ))
#define EXPECT_DEATH_MATCHES(matcher, ...)        \
    assert(( ExpectDeath(DeathCause::VerboseAbort, #__VA_ARGS__, [&]() { __VA_ARGS__; }, matcher) ))
#define EXPECT_STD_TERMINATE(...)                 \
    assert(  ExpectDeath(DeathCause::StdTerminate, #__VA_ARGS__, __VA_ARGS__)  )

#if _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_DEBUG
#define TEST_LIBCPP_ASSERT_FAILURE(expr, message) \
    assert(( ExpectDeath(DeathCause::VerboseAbort, #expr, [&]() { (void)(expr); }, AssertionInfoMatcher(message)) ))
#else
#define TEST_LIBCPP_ASSERT_FAILURE(expr, message) \
    assert(( ExpectDeath(DeathCause::Trap,         #expr, [&]() { (void)(expr); }) ))
#endif // _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_DEBUG

// clang-format on

#endif // TEST_SUPPORT_CHECK_ASSERTION_H
