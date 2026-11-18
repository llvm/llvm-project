//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of sig2str.
///
//===----------------------------------------------------------------------===//

#include "src/signal/sig2str.h"
#include "hdr/signal_macros.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/StringUtil/signal_to_string.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sig2str, (int signum, char *str)) {
  if (str == nullptr)
    return -1;

  char buf[SIG2STR_MAX];
  cpp::string_view name = get_signal_name(signum, {buf, sizeof(buf)});
  if (name.empty())
    return -1;

  for (size_t i = 0; i < name.size(); ++i)
    str[i] = name[i];
  str[name.size()] = '\0';
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
