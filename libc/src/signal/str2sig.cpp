//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of str2sig.
///
//===----------------------------------------------------------------------===//

#include "src/signal/str2sig.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/StringUtil/signal_to_string.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, str2sig,
                   (const char *__restrict str, int *__restrict pnum)) {
  if (str == nullptr || pnum == nullptr)
    return -1;

  auto opt = get_signal_number(cpp::string_view(str));
  if (!opt.has_value())
    return -1;

  *pnum = opt.value();
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
