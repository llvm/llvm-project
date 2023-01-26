//===-- Implementation of strsignal
//----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strsignal.h"
#include "src/__support/StringUtil/signal_to_string.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strsignal, (int sig_num)) {
  return const_cast<char *>(get_signal_string(sig_num).data());
}

} // namespace __llvm_libc
