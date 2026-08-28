//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Collection of utils for asctime.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_ASCTIME_UTILS_H
#define LLVM_LIBC_SRC_TIME_ASCTIME_UTILS_H

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_tm.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/printf_core/writer.h"
#include "src/time/strftime_core/strftime_main.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace time_utils {

LIBC_INLINE ErrorOr<char *> asctime(const tm *timeptr, char *buffer,
                                    size_t buffer_length) {
  if (timeptr == nullptr || buffer == nullptr)
    return cpp::unexpected(EINVAL);
  if (timeptr->tm_wday < 0 ||
      timeptr->tm_wday > (time_constants::DAYS_PER_WEEK - 1))
    return cpp::unexpected(EINVAL);
  if (timeptr->tm_mon < 0 ||
      timeptr->tm_mon > (time_constants::MONTHS_PER_YEAR - 1))
    return cpp::unexpected(EINVAL);

  printf_core::DropOverflowBuffer wb(buffer,
                                     buffer_length > 0 ? buffer_length - 1 : 0);
  printf_core::Writer writer(wb);

  auto res = strftime_core::strftime_main(&writer, "%a %b %e %T %Y\n", timeptr);
  if (!res.has_value())
    return cpp::unexpected(res.error());

  if (res.value() >= buffer_length)
    return cpp::unexpected(TIME_OVERFLOW);

  buffer[res.value()] = '\0';
  return buffer;
}

} // namespace time_utils
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_ASCTIME_UTILS_H
