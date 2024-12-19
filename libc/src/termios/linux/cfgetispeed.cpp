//===-- Linux implementation of cfgetispeed -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/termios/cfgetispeed.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <termios.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(speed_t, cfgetispeed, (const struct termios *t)) {
  return t->c_cflag & CBAUD;
}

} // namespace LIBC_NAMESPACE_DECL
