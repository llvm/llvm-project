//===-- Implementation of timezone functions ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/linux/timezone.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace timezone {

tzset *get_tzset(File *file) {
  static tzset result;
  (void)file;

  return &result;
}

} // namespace timezone
} // namespace LIBC_NAMESPACE_DECL
