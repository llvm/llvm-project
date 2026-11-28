//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for syslog_internal.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/syslog_macros.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_tm.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/arg_list.h"
#include "src/__support/libc_errno.h"
#include "src/__support/printf_core/core_structs.h"
#include "src/__support/printf_core/writer.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/syslog/linux/syslog_internal.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

#include <stdarg.h>

using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::printf_core::overflow_write_flush_to_sink;
using LIBC_NAMESPACE::syslog_internal::detail::LogState;

struct CaptureBuffer {
  char data[2048] = {};
  size_t length = 0;

  static int write_hook(string_view str, void *state) {
    auto *capture = static_cast<CaptureBuffer *>(state);
    size_t remaining = sizeof(capture->data) - capture->length;
    size_t to_copy = str.size() < remaining ? str.size() : remaining;
    LIBC_NAMESPACE::inline_memcpy(capture->data + capture->length, str.data(),
                                  to_copy);
    capture->length += to_copy;
    return LIBC_NAMESPACE::printf_core::WRITE_OK;
  }

  void clear() { length = 0; }
  string_view view() const { return string_view(data, length); }
};

static bool format_to_capture(const LogState &state, int priority,
                              const tm &timestamp, pid_t pid,
                              const char *default_ident, CaptureBuffer &capture,
                              const char *format, ...) {
  capture.clear();
  char scratch[64];
  LIBC_NAMESPACE::printf_core::Writer writer =
      LIBC_NAMESPACE::printf_core::make_writer(
          scratch, sizeof(scratch),
          &overflow_write_flush_to_sink<char, CaptureBuffer::write_hook>,
          &capture);
  va_list vlist;
  va_start(vlist, format);
  LIBC_NAMESPACE::internal::ArgList args(vlist);
  va_end(vlist);
  bool wrote = LIBC_NAMESPACE::syslog_internal::detail::write_log_message(
      writer, state, priority, format, args, timestamp, pid, default_ident);
  if (wrote)
    writer.get_write_buffer().flush_to_sink<CaptureBuffer::write_hook>(
        &capture);
  return wrote;
}

using LlvmLibcSyslogInternalTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSyslogInternalTest, DefaultStateUsesLogUserAndDefaultIdent) {
  LogState state;
  CaptureBuffer capture;
  tm time = {};
  time.tm_year = 126;
  time.tm_mon = 10;
  time.tm_mday = 14;
  time.tm_hour = 1;
  time.tm_min = 2;
  time.tm_sec = 3;

  ASSERT_TRUE(format_to_capture(state, LOG_INFO, time, 1234, "myprog", capture,
                                "hello %d", 42));
  ASSERT_EQ(capture.view(),
            string_view("<14>Nov 14 01:02:03 myprog: hello 42\n"));
}

TEST_F(LlvmLibcSyslogInternalTest, ConfigureWithPidAndIdent) {
  LogState state;
  CaptureBuffer capture;
  tm time = {};
  time.tm_year = 126;
  time.tm_mon = 10;
  time.tm_mday = 14;
  time.tm_hour = 1;
  time.tm_min = 2;
  time.tm_sec = 3;

  state.configure("prog", LOG_PID, LOG_USER);
  ASSERT_TRUE(format_to_capture(state, LOG_INFO, time, 1234, "ignored", capture,
                                "hello %d", 42));
  ASSERT_EQ(capture.view(),
            string_view("<14>Nov 14 01:02:03 prog[1234]: hello 42\n"));
}

TEST_F(LlvmLibcSyslogInternalTest, WithoutPidPadsSingleDigitDay) {
  LogState state;
  CaptureBuffer capture;
  tm time = {};
  time.tm_year = 126;
  time.tm_mon = 0;
  time.tm_mday = 5;
  time.tm_hour = 23;
  time.tm_min = 59;
  time.tm_sec = 58;

  state.configure("prog", 0, LOG_USER);
  ASSERT_TRUE(format_to_capture(state, LOG_INFO, time, 1234, nullptr, capture,
                                "%s", "msg"));
  ASSERT_EQ(capture.view(), string_view("<14>Jan  5 23:59:58 prog: msg\n"));
}

TEST_F(LlvmLibcSyslogInternalTest, FacilitySelectionAndOverride) {
  LogState state;
  CaptureBuffer capture;
  tm time = {};
  time.tm_year = 126;
  time.tm_mon = 10;
  time.tm_mday = 14;
  time.tm_hour = 1;
  time.tm_min = 2;
  time.tm_sec = 3;

  state.configure("prog", 0, LOG_LOCAL0);
  ASSERT_TRUE(
      format_to_capture(state, LOG_ERR, time, 0, nullptr, capture, "default"));
  ASSERT_EQ(capture.view(),
            string_view("<131>Nov 14 01:02:03 prog: default\n"));

  // Explicit facility in priority overrides the openlog default.
  ASSERT_TRUE(format_to_capture(state, LOG_LOCAL7 | LOG_DEBUG, time, 0, nullptr,
                                capture, "explicit"));
  ASSERT_EQ(capture.view(),
            string_view("<191>Nov 14 01:02:03 prog: explicit\n"));

  // Passing facility == 0 preserves the current default facility.
  state.configure("prog2", LOG_NOWAIT, 0);
  ASSERT_TRUE(
      format_to_capture(state, LOG_ERR, time, 0, nullptr, capture, "kept"));
  ASSERT_EQ(capture.view(), string_view("<131>Nov 14 01:02:03 prog2: kept\n"));
}

TEST_F(LlvmLibcSyslogInternalTest, ResetClearsAttributesAndKeepsMask) {
  LogState state;
  CaptureBuffer capture;
  tm time = {};
  time.tm_year = 126;
  time.tm_mon = 10;
  time.tm_mday = 14;
  time.tm_hour = 1;
  time.tm_min = 2;
  time.tm_sec = 3;

  state.configure("prog", LOG_PID, LOG_LOCAL0);
  ASSERT_EQ(state.update_mask(LOG_MASK(LOG_ERR)), LOG_UPTO(LOG_DEBUG));
  state.reset();

  ASSERT_EQ(state.update_mask(0), LOG_MASK(LOG_ERR));
  ASSERT_TRUE(format_to_capture(state, LOG_ERR, time, 99, nullptr, capture,
                                "after close"));
  ASSERT_EQ(capture.view(), string_view("<11>Nov 14 01:02:03 : after close\n"));
}

TEST_F(LlvmLibcSyslogInternalTest, UpdateMaskFiltersMessages) {
  LogState state;
  CaptureBuffer capture;
  tm time = {};
  time.tm_year = 126;
  time.tm_mon = 10;
  time.tm_mday = 14;
  time.tm_hour = 1;
  time.tm_min = 2;
  time.tm_sec = 3;

  state.configure("prog", 0, LOG_USER);
  ASSERT_EQ(state.update_mask(LOG_UPTO(LOG_ERR)), LOG_UPTO(LOG_DEBUG));

  ASSERT_FALSE(format_to_capture(state, LOG_DEBUG, time, 0, nullptr, capture,
                                 "dropped"));
  ASSERT_EQ(capture.length, size_t(0));
  ASSERT_FALSE(format_to_capture(state, LOG_LOCAL0 | LOG_INFO, time, 0, nullptr,
                                 capture, "also dropped"));
  ASSERT_EQ(capture.length, size_t(0));

  ASSERT_TRUE(format_to_capture(state, LOG_LOCAL0 | LOG_ERR, time, 0, nullptr,
                                capture, "delivered"));
  ASSERT_EQ(capture.view(),
            string_view("<131>Nov 14 01:02:03 prog: delivered\n"));
}

TEST_F(LlvmLibcSyslogInternalTest, PreservesExistingTrailingNewline) {
  LogState state;
  CaptureBuffer capture;
  tm time = {};
  time.tm_year = 126;
  time.tm_mon = 10;
  time.tm_mday = 14;
  time.tm_hour = 1;
  time.tm_min = 2;
  time.tm_sec = 3;

  state.configure("prog", 0, LOG_USER);
  ASSERT_TRUE(format_to_capture(state, LOG_INFO, time, 0, nullptr, capture,
                                "already ended\n"));
  ASSERT_EQ(capture.view(),
            string_view("<14>Nov 14 01:02:03 prog: already ended\n"));
}

TEST_F(LlvmLibcSyslogInternalTest, FlushesMessagesLongerThanBuffer) {
  LogState state;
  CaptureBuffer capture;
  tm time = {};
  time.tm_year = 126;
  time.tm_mon = 10;
  time.tm_mday = 14;
  time.tm_hour = 1;
  time.tm_min = 2;
  time.tm_sec = 3;

  state.configure("prog", 0, LOG_USER);
  ASSERT_TRUE(format_to_capture(state, LOG_INFO, time, 0, nullptr, capture,
                                "%120s", "end"));
  ASSERT_EQ(capture.length, size_t(147));
  ASSERT_EQ(capture.view().substr(0, 26),
            string_view("<14>Nov 14 01:02:03 prog: "));
  ASSERT_EQ(capture.view().substr(capture.length - 4), string_view("end\n"));
}

TEST_F(LlvmLibcSyslogInternalTest, EscapedPercentMIsLiteral) {
  LogState state;
  CaptureBuffer capture;
  tm time = {};
  time.tm_year = 126;
  time.tm_mon = 10;
  time.tm_mday = 14;
  time.tm_hour = 1;
  time.tm_min = 2;
  time.tm_sec = 3;

  state.configure("prog", 0, LOG_USER);
  libc_errno = EINVAL;
  ASSERT_TRUE(format_to_capture(state, LOG_INFO, time, 0, nullptr, capture,
                                "literal %%m"));
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_EQ(capture.view(),
            string_view("<14>Nov 14 01:02:03 prog: literal %m\n"));
}

#ifndef LIBC_COPT_PRINTF_DISABLE_STRERROR
TEST_F(LlvmLibcSyslogInternalTest, PercentMFormatsErrnoAndConvertsNoArgs) {
  LogState state;
  CaptureBuffer capture;
  tm time = {};
  time.tm_year = 126;
  time.tm_mon = 10;
  time.tm_mday = 14;
  time.tm_hour = 1;
  time.tm_min = 2;
  time.tm_sec = 3;

  state.configure("prog", 0, LOG_USER);
  libc_errno = EINVAL;
  ASSERT_TRUE(format_to_capture(state, LOG_INFO, time, 0, nullptr, capture,
                                "%s: %m (%d) [%m]", "open", 7));
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_EQ(capture.view(),
            string_view("<14>Nov 14 01:02:03 prog: open: Invalid argument (7) "
                        "[Invalid argument]\n"));
}

TEST_F(LlvmLibcSyslogInternalTest, PercentMWithPositionalArguments) {
  LogState state;
  CaptureBuffer capture;
  tm time = {};
  time.tm_year = 126;
  time.tm_mon = 10;
  time.tm_mday = 14;
  time.tm_hour = 1;
  time.tm_min = 2;
  time.tm_sec = 3;

  state.configure("prog", 0, LOG_USER);
  libc_errno = EINVAL;
  ASSERT_TRUE(format_to_capture(state, LOG_INFO, time, 0, nullptr, capture,
                                "%2$d: %m (%1$s)", "open", 7));
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_EQ(capture.view(),
            string_view("<14>Nov 14 01:02:03 prog: 7: Invalid argument "
                        "(open)\n"));
}
#endif // LIBC_COPT_PRINTF_DISABLE_STRERROR
