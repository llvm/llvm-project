//===-- Implementation of timezone functions ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/timezone.h"
#include "src/__support/CPP/limits.h" // INT_MIN, INT_MAX
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/time/time_utils.h"

#define BUF_SIZE 1024

namespace LIBC_NAMESPACE_DECL {
namespace timezone {

using LIBC_NAMESPACE::time_utils::TimeConstants;

#include <stdio.h>
#include <stdlib.h>

int get_timezone_offset(char *timezone) {
  int offset = 0;
  LIBC_NAMESPACE::cpp::string_view tz(timezone);

  if (tz.starts_with("America")) {
    if (tz.ends_with("San_Francisco")) {
      offset = -8;
    }

    if (tz.ends_with("Chicago")) {
      offset = -4;
    }

    if (tz.ends_with("New_York")) {
      offset = -5;
    }
  }

  if (tz.starts_with("Europe")) {
    offset = 1;

    if (tz.ends_with("Lisbon")) {
      offset = 0;
    }

    if (tz.ends_with("Moscow")) {
      offset = 2;
    }
  }

  return offset;
}

} // namespace timezone
} // namespace LIBC_NAMESPACE_DECL
