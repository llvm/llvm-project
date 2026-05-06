//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of regerror.
///
//===----------------------------------------------------------------------===//

#include "src/regex/regerror.h"

#include "hdr/regex_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, regerror,
                   (int errcode, const regex_t *__restrict preg,
                    char *__restrict errbuf, size_t errbuf_size)) {
  (void)preg; // preg is reserved for implementation-specific messages.

  const char *msg;
  switch (errcode) {
  case 0:
    msg = "Success";
    break;
  case REG_NOMATCH:
    msg = "No match";
    break;
  case REG_BADPAT:
    msg = "Invalid regular expression";
    break;
  case REG_ECOLLATE:
    msg = "Invalid collating element";
    break;
  case REG_ECTYPE:
    msg = "Invalid character class";
    break;
  case REG_EESCAPE:
    msg = "Trailing backslash";
    break;
  case REG_ESUBREG:
    msg = "Invalid backreference";
    break;
  case REG_EBRACK:
    msg = "Missing ']'";
    break;
  case REG_EPAREN:
    msg = "Missing ')'";
    break;
  case REG_EBRACE:
    msg = "Missing '}'";
    break;
  case REG_BADBR:
    msg = "Invalid repetition count";
    break;
  case REG_ERANGE:
    msg = "Invalid range end";
    break;
  case REG_ESPACE:
    msg = "Out of memory";
    break;
  case REG_BADRPT:
    msg = "Invalid preceding expression";
    break;
  default:
    msg = "Unknown error";
    break;
  }

  size_t msg_len = internal::string_length(msg) + 1; // include NUL

  if (errbuf_size > 0 && errbuf) {
    size_t copy_len = msg_len < errbuf_size ? msg_len : errbuf_size;
    inline_memcpy(errbuf, msg, copy_len - 1);
    errbuf[copy_len - 1] = '\0';
  }
  // POSIX requires returning the size needed to hold the full NUL-terminated
  // string, even if it was truncated in the buffer.
  return msg_len;
}

} // namespace LIBC_NAMESPACE_DECL
