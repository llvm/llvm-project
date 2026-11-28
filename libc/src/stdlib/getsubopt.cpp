//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getsubopt.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/getsubopt.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getsubopt,
                   (char **__restrict optionp, char *const *__restrict tokens,
                    char **__restrict valuep)) {
  LIBC_CRASH_ON_NULLPTR(optionp);
  LIBC_CRASH_ON_NULLPTR(*optionp);
  LIBC_CRASH_ON_NULLPTR(tokens);
  LIBC_CRASH_ON_NULLPTR(valuep);

  char *cur = *optionp;
  // If the option string is empty, we are done.
  if (*cur == '\0')
    return -1;

  char *token_start = cur;
  char *val = nullptr;
  size_t key_len = 0;
  bool found_equal = false;

  // Scan until comma or end of string.
  while (*cur != '\0' && *cur != ',') {
    if (*cur == '=' && !found_equal) {
      found_equal = true;
      key_len = static_cast<size_t>(cur - token_start);
      val = cur + 1;
    }
    ++cur;
  }

  if (!found_equal)
    key_len = static_cast<size_t>(cur - token_start);

  if (*cur == ',') {
    *cur = '\0';
    *optionp = cur + 1;
  } else {
    *optionp = cur;
  }

  cpp::string_view key(token_start, key_len);
  for (int idx = 0; tokens[idx] != nullptr; ++idx) {
    if (key == tokens[idx]) {
      *valuep = val;
      return idx;
    }
  }

  // Token was not found in the tokens array; valuep points to the unmatched
  // suboption string.
  *valuep = token_start;
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
