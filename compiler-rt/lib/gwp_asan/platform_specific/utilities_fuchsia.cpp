//===-- utilities_fuchsia.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/utilities.h"

#include <alloca.h>
#include <inttypes.h>
#include <string.h>
#include <zircon/sanitizer.h>

namespace gwp_asan {
void die(const char *Message) {
  __sanitizer_log_write(Message, strlen(Message));
  __builtin_trap();
}

void dieWithErrorCode(const char *Message, int64_t ErrorCode) {
  size_t buffer_size = strlen(Message) + 48;
  char *buffer = static_cast<char *>(alloca(strlen(Message) + 48));
  snprintf(buffer, buffer_size, "%s (Error Code: %" PRId64 ")", Message,
           ErrorCode);
  __sanitizer_log_write(buffer, strlen(buffer));
  __builtin_trap();
}
} // namespace gwp_asan
