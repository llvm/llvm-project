//===-- common.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.h"
#include "atomic_helpers.h"
#include "string_utils.h"

namespace scudo {

#if !defined(SCUDO_PAGE_SIZE)
uptr PageSizeCached = 0;
uptr PageSizeLogCached = 0;

// This must be called in the init path or there could be a race if multiple
// threads try to set the cached values.
uptr getPageSizeSlow() {
  PageSizeCached = getPageSize();
  CHECK_NE(PageSizeCached, 0);
  PageSizeLogCached = getLog2(PageSizeCached);
  return PageSizeCached;
}
#endif

} // namespace scudo
