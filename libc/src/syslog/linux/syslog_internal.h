//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal state and formatting shared by the Linux syslog entrypoints.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYSLOG_LINUX_SYSLOG_INTERNAL_H
#define LLVM_LIBC_SRC_SYSLOG_LINUX_SYSLOG_INTERNAL_H

#include "hdr/syslog_macros.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/struct_tm.h"
#include "src/__support/arg_list.h"
#include "src/__support/macros/config.h"
#include "src/__support/printf_core/writer.h"

namespace LIBC_NAMESPACE_DECL {
namespace syslog_internal {

// Unlocked state and formatting helpers (exposed here for unit testing).
// Callers operating on shared state must synchronize externally.
namespace detail {

inline constexpr int LEVEL_MASK = 0x07;
inline constexpr int FACILITY_MASK = 0x3f8;

struct LogState {
  const char *ident = nullptr;
  int options = 0;
  int facility = LOG_USER;
  int mask = LOG_UPTO(LOG_DEBUG);

  LIBC_INLINE void configure(const char *new_ident, int new_options,
                             int new_facility) {
    ident = new_ident;
    options = new_options;
    if (new_facility != 0)
      facility = new_facility & FACILITY_MASK;
  }

  LIBC_INLINE void reset() {
    ident = nullptr;
    options = 0;
    facility = LOG_USER;
  }

  LIBC_INLINE int update_mask(int new_mask) {
    int old_mask = mask;
    if (new_mask != 0)
      mask = new_mask;
    return old_mask;
  }

  LIBC_INLINE bool allows_priority(int priority) const {
    return (mask & LOG_MASK(priority & LEVEL_MASK)) != 0;
  }
};

// Formats a log message into `writer` if `priority` is permitted by `state`.
// Uses `default_ident` when `state.ident` is null. Returns true if a message
// was written, false if filtered or on error.
bool write_log_message(
    printf_core::Writer<printf_core::OverflowMode::CALLBACK> &writer,
    const LogState &state, int priority, const char *__restrict format,
    internal::ArgList &args, const tm &timestamp, pid_t pid,
    const char *default_ident);

} // namespace detail

void open_log(const char *ident, int options, int facility);

void close_log();

int set_log_mask(int mask);

void log_message(int priority, const char *__restrict format,
                 internal::ArgList &args);

} // namespace syslog_internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYSLOG_LINUX_SYSLOG_INTERNAL_H
