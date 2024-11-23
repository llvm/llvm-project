//===-- Implementation of timezone functions ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/time_utils.h"
#include "src/__support/common.h"
#include "src/time/timezone.h"

namespace LIBC_NAMESPACE_DECL {
namespace timezone {

using LIBC_NAMESPACE::time_utils::TimeConstants;

int get_timezone_offset(char *timezone) {
  (void)timezone;
  return 0;
}

} // namespace timezone
} // namespace LIBC_NAMESPACE_DECL
