//===-- utilities_fuchsia.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/utilities.h"

#include <alloca.h>
#include <stdio.h>
#include <string.h>
#include <zircon/sanitizer.h>
#include <zircon/status.h>

namespace gwp_asan {
void die(const char *Message) {
  __sanitizer_log_write(Message, strlen(Message));
  __builtin_trap();
}

void dieWithErrorCode(const char *Message, int64_t ErrorCode) {
  const char *error_str =
      _zx_status_get_string(static_cast<zx_status_t>(ErrorCode));
  size_t buffer_size = strlen(Message) + 32 + strlen(error_str);
  char *buffer = static_cast<char *>(alloca(buffer_size));
  snprintf(buffer, buffer_size, "%s (Error Code: %s)", Message, error_str);
  __sanitizer_log_write(buffer, strlen(buffer));
  __builtin_trap();
}
} // namespace gwp_asan
