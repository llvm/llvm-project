//===-- Implementation of a class for mapping signals to strings ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/StringUtil/signal_to_string.h"

#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/stringstream.h"
#include "src/__support/StringUtil/message_mapper.h"
#include "src/__support/integer_to_string.h"

#include <signal.h>
#include <stddef.h>

namespace __llvm_libc {
namespace internal {

constexpr size_t max_buff_size() {
  constexpr size_t base_str_len = sizeof("Real-time signal");
  constexpr size_t max_num_len =
      __llvm_libc::IntegerToString::dec_bufsize<int>();
  // the buffer should be able to hold "Real-time signal" + ' ' + num_str
  return (base_str_len + 1 + max_num_len) * sizeof(char);
}

// This is to hold signal strings that have to be custom built. It may be
// rewritten on every call to strsignal (or other signal to string function).
constexpr size_t SIG_BUFFER_SIZE = max_buff_size();
thread_local char signal_buffer[SIG_BUFFER_SIZE];

constexpr MsgMapping raw_sig_array[] = {
    MsgMapping(SIGHUP, "Hangup"), MsgMapping(SIGINT, "Interrupt"),
    MsgMapping(SIGQUIT, "Quit"), MsgMapping(SIGILL, "Illegal instruction"),
    MsgMapping(SIGTRAP, "Trace/breakpoint trap"),
    MsgMapping(SIGABRT, "Aborted"), MsgMapping(SIGBUS, "Bus error"),
    MsgMapping(SIGFPE, "Floating point exception"),
    MsgMapping(SIGKILL, "Killed"), MsgMapping(SIGUSR1, "User defined signal 1"),
    MsgMapping(SIGSEGV, "Segmentation fault"),
    MsgMapping(SIGUSR2, "User defined signal 2"),
    MsgMapping(SIGPIPE, "Broken pipe"), MsgMapping(SIGALRM, "Alarm clock"),
    MsgMapping(SIGTERM, "Terminated"),
    // SIGSTKFLT (may not exist)
    MsgMapping(SIGCHLD, "Child exited"), MsgMapping(SIGCONT, "Continued"),
    MsgMapping(SIGSTOP, "Stopped (signal)"), MsgMapping(SIGTSTP, "Stopped"),
    MsgMapping(SIGTTIN, "Stopped (tty input)"),
    MsgMapping(SIGTTOU, "Stopped (tty output)"),
    MsgMapping(SIGURG, "Urgent I/O condition"),
    MsgMapping(SIGXCPU, "CPU time limit exceeded"),
    MsgMapping(SIGXFSZ, "File size limit exceeded"),
    MsgMapping(SIGVTALRM, "Virtual timer expired"),
    MsgMapping(SIGPROF, "Profiling timer expired"),
    MsgMapping(SIGWINCH, "Window changed"), MsgMapping(SIGPOLL, "I/O possible"),
    // SIGPWR (may not exist)
    MsgMapping(SIGSYS, "Bad system call"),

#ifdef SIGSTKFLT
    MsgMapping(SIGSTKFLT, "Stack fault"), // unused
#endif
#ifdef SIGPWR
    MsgMapping(SIGPWR, "Power failure"), // ignored
#endif
};

// Since the string_mappings array is a map from signal numbers to their
// corresponding strings, we have to have an array large enough we can use the
// signal numbers as indexes. The highest signal is SIGSYS at 31, so an array of
// 32 elements will be large enough to hold all of them.
constexpr size_t SIG_ARRAY_SIZE = 32;

constexpr size_t RAW_ARRAY_LEN = sizeof(raw_sig_array) / sizeof(MsgMapping);
constexpr size_t TOTAL_STR_LEN = total_str_len(raw_sig_array, RAW_ARRAY_LEN);

static constexpr MessageMapper<SIG_ARRAY_SIZE, TOTAL_STR_LEN>
    signal_mapper(raw_sig_array, RAW_ARRAY_LEN);

cpp::string_view build_signal_string(int sig_num, cpp::span<char> buffer) {
  cpp::string_view base_str;
  if (sig_num >= SIGRTMIN && sig_num <= SIGRTMAX) {
    base_str = cpp::string_view("Real-time signal");
    sig_num -= SIGRTMIN;
  } else {
    base_str = cpp::string_view("Unknown signal");
  }

  // if the buffer can't hold "Unknown signal" + ' ' + num_str, then just
  // return "Unknown signal".
  if (buffer.size() <
      (base_str.size() + 1 + IntegerToString::dec_bufsize<int>()))
    return base_str;

  cpp::StringStream buffer_stream(
      {const_cast<char *>(buffer.data()), buffer.size()});
  buffer_stream << base_str << ' ' << sig_num << '\0';
  return buffer_stream.str();
}

} // namespace internal

cpp::string_view get_signal_string(int sig_num) {
  return get_signal_string(
      sig_num, {internal::signal_buffer, internal::SIG_BUFFER_SIZE});
}

cpp::string_view get_signal_string(int sig_num, cpp::span<char> buffer) {
  auto opt_str = internal::signal_mapper.get_str(sig_num);
  if (opt_str)
    return *opt_str;
  else
    return internal::build_signal_string(sig_num, buffer);
}

} // namespace __llvm_libc
