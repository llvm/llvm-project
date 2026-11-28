//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of internal state and formatting for syslog.
///
//===----------------------------------------------------------------------===//

#include "src/syslog/linux/syslog_internal.h"

#include "hdr/syslog_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_timespec.h"
#include "hdr/types/struct_tm.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getpid.h"
#include "src/__support/arg_list.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/printf_core/core_structs.h"
#include "src/__support/printf_core/printf_main.h"
#include "src/__support/printf_core/writer.h"
#include "src/__support/threads/mutex.h"
#include "src/__support/time/clock_gettime.h"
#include "src/errno/program_invocation_short_name.h"
#include "src/time/strftime_core/strftime_main.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace syslog_internal {

namespace {

constexpr size_t BUFFER_SIZE = 1024;

Mutex log_mutex(/*is_priority_inherit=*/false, /*is_recursive=*/false,
                /*is_robust=*/false, /*is_pshared=*/false);
LIBC_CONSTINIT detail::LogState log_state;

LIBC_INLINE int stderr_write_hook(cpp::string_view str, void *) {
  write_to_stderr(str);
  return printf_core::WRITE_OK;
}

} // namespace

namespace detail {

bool write_log_message(
    printf_core::Writer<printf_core::OverflowMode::CALLBACK> &writer,
    const LogState &state, int priority, const char *__restrict format,
    internal::ArgList &args, const tm &timestamp, pid_t pid,
    const char *default_ident) {
  if (!state.allows_priority(priority))
    return false;
  if ((priority & FACILITY_MASK) == 0)
    priority |= state.facility;

  const char *ident = state.ident;
  if (ident == nullptr)
    ident = default_ident;
  if (ident == nullptr)
    ident = "";

  writer.write('<');
  const IntegerToString<int> priority_str(priority);
  writer.write(priority_str.view());
  writer.write('>');

  // RFC 3164 timestamp format: abbreviated month name, space-padded day of
  // month, and 24-hour HH:MM:SS time
  // (e.g. "Nov 14 01:02:03 " or "Jan  5 23:59:58 ").
  if (!strftime_core::strftime_main(&writer, "%b %e %T ", &timestamp)
           .has_value())
    return false;

  writer.write(ident);
  if (state.options & LOG_PID) {
    writer.write('[');
    const IntegerToString<pid_t> pid_str(pid);
    writer.write(pid_str.view());
    writer.write(']');
  }
  writer.write(cpp::string_view(": "));

  if (!printf_core::printf_main(&writer, format, args).has_value())
    return false;

  const auto &wb = writer.get_write_buffer();
  if (wb.buff_cur == 0 || wb.buff[wb.buff_cur - 1] != '\n')
    writer.write('\n');

  return true;
}

} // namespace detail

void open_log(const char *ident, int options, int facility) {
  log_mutex.lock();
  log_state.configure(ident, options, facility);
  log_mutex.unlock();
}

void close_log() {
  log_mutex.lock();
  log_state.reset();
  log_mutex.unlock();
}

int set_log_mask(int mask) {
  log_mutex.lock();
  int old_mask = log_state.update_mask(mask);
  log_mutex.unlock();
  return old_mask;
}

void log_message(int priority, const char *__restrict format,
                 internal::ArgList &args) {
  log_mutex.lock();
  if (log_state.allows_priority(priority)) {
    timespec now = {};
    internal::clock_gettime(CLOCK_REALTIME, &now);
    tm local_time = {};
    if (time_utils::localtime_internal(&now.tv_sec, &local_time).has_value()) {
      pid_t pid = (log_state.options & LOG_PID) ? linux_syscalls::getpid() : 0;
      char buffer[BUFFER_SIZE];
      printf_core::Writer writer = printf_core::make_writer(
          buffer, sizeof(buffer),
          &printf_core::overflow_write_flush_to_sink<char, stderr_write_hook>);
      if (detail::write_log_message(writer, log_state, priority, format, args,
                                    local_time, pid,
                                    program_invocation_short_name))
        writer.get_write_buffer().flush_to_sink<stderr_write_hook>();
    }
  }
  log_mutex.unlock();
}

} // namespace syslog_internal
} // namespace LIBC_NAMESPACE_DECL
