//===-- Unittests for strsignal -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strsignal.h"
#include "test/UnitTest/Test.h"

#include <signal.h>

TEST(LlvmLibcStrSignalTest, KnownSignals) {
  ASSERT_STREQ(__llvm_libc::strsignal(1), "Hangup");

  const char *message_array[] = {
      "Unknown signal 0", // unknown
      "Hangup",
      "Interrupt",
      "Quit",
      "Illegal instruction",
      "Trace/breakpoint trap",
      "Aborted",
      "Bus error",
      "Floating point exception",
      "Killed",
      "User defined signal 1",
      "Segmentation fault",
      "User defined signal 2",
      "Broken pipe",
      "Alarm clock",
      "Terminated",
      "Stack fault",
      "Child exited",
      "Continued",
      "Stopped (signal)",
      "Stopped",
      "Stopped (tty input)",
      "Stopped (tty output)",
      "Urgent I/O condition",
      "CPU time limit exceeded",
      "File size limit exceeded",
      "Virtual timer expired",
      "Profiling timer expired",
      "Window changed",
      "I/O possible",
      "Power failure",
      "Bad system call",
  };

  // There are supposed to be 32 of these, but sometimes SIGRTMIN is shifted to
  // reserve some.
  const char *rt_message_array[] = {
      "Real-time signal 0",  "Real-time signal 1",  "Real-time signal 2",
      "Real-time signal 3",  "Real-time signal 4",  "Real-time signal 5",
      "Real-time signal 6",  "Real-time signal 7",  "Real-time signal 8",
      "Real-time signal 9",  "Real-time signal 10", "Real-time signal 11",
      "Real-time signal 12", "Real-time signal 13", "Real-time signal 14",
      "Real-time signal 15", "Real-time signal 16", "Real-time signal 17",
      "Real-time signal 18", "Real-time signal 19", "Real-time signal 20",
      "Real-time signal 21", "Real-time signal 22", "Real-time signal 23",
      "Real-time signal 24", "Real-time signal 25", "Real-time signal 26",
      "Real-time signal 27", "Real-time signal 28", "Real-time signal 29",
      "Real-time signal 30", "Real-time signal 31", "Real-time signal 32",
  };

  for (size_t i = 0; i < (sizeof(message_array) / sizeof(char *)); ++i) {
    EXPECT_STREQ(__llvm_libc::strsignal(i), message_array[i]);
  }

  for (size_t i = 0; i < SIGRTMAX - SIGRTMIN; ++i) {
    EXPECT_STREQ(__llvm_libc::strsignal(i + SIGRTMIN), rt_message_array[i]);
  }
}

TEST(LlvmLibcStrsignalTest, UnknownSignals) {
  ASSERT_STREQ(__llvm_libc::strsignal(-1), "Unknown signal -1");
  ASSERT_STREQ(__llvm_libc::strsignal(65), "Unknown signal 65");
  ASSERT_STREQ(__llvm_libc::strsignal(2147483647), "Unknown signal 2147483647");
  ASSERT_STREQ(__llvm_libc::strsignal(-2147483648),
               "Unknown signal -2147483648");
}
