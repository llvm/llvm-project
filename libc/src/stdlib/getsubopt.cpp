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

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getsubopt,
                   (char **optionp, char *const *tokens, char **valuep)) {
  if (optionp == nullptr || *optionp == nullptr || tokens == nullptr ||
      valuep == nullptr)
    return -1;

  char *cur = *optionp;
  // If the option string is empty, we are done.
  if (*cur == '\0')
    return -1;

  char *token_start = cur;
  char *val = nullptr;

  // Scan until comma or end of string.
  while (*cur != '\0' && *cur != ',') {
    if (*cur == '=' && val == nullptr) {
      *cur = '\0';
      val = cur + 1;
    }
    ++cur;
  }

  if (*cur == ',') {
    *cur = '\0';
    *optionp = cur + 1;
  } else {
    *optionp = cur;
  }

  *valuep = val;

  cpp::string_view key(token_start);
  for (int idx = 0; tokens[idx] != nullptr; ++idx) {
    if (key == tokens[idx])
      return idx;
  }

  // Token was not found in the tokens array; valuep points to the unmatched token.
  *valuep = token_start;
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
