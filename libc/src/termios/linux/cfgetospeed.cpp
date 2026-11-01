//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of cfgetospeed.
///
//===----------------------------------------------------------------------===//

#include "src/termios/cfgetospeed.h"
#include "hdr/types/speed_t.h"
#include "hdr/types/struct_termios.h"
#include "src/__support/common.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(speed_t, cfgetospeed, (const termios *t)) {
  LIBC_CRASH_ON_NULLPTR(t);
  return t->c_ospeed;
}

} // namespace LIBC_NAMESPACE_DECL
