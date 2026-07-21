//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of cfgetispeed.
///
//===----------------------------------------------------------------------===//

#include "src/termios/cfgetispeed.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include "hdr/types/speed_t.h"
#include "hdr/types/struct_termios.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(speed_t, cfgetispeed, (const struct termios *t)) {
  return t->c_ispeed;
}

} // namespace LIBC_NAMESPACE_DECL
