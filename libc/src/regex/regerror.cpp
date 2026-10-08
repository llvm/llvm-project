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
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, regerror,
                   (int errcode, const regex_t *__restrict preg,
                    char *__restrict errbuf, size_t errbuf_size)) {
  (void)preg; // preg is reserved for implementation-specific messages.

  cpp::string_view msg = [errcode]() -> const char * {
    switch (errcode) {
    case 0:
      return "Success";
    case REG_NOMATCH:
      return "No match";
    case REG_BADPAT:
      return "Invalid regular expression";
    case REG_ECOLLATE:
      return "Invalid collating element";
    case REG_ECTYPE:
      return "Invalid character class";
    case REG_EESCAPE:
      return "Trailing backslash";
    case REG_ESUBREG:
      return "Invalid backreference";
    case REG_EBRACK:
      return "Missing ']'";
    case REG_EPAREN:
      return "Missing ')'";
    case REG_EBRACE:
      return "Missing '}'";
    case REG_BADBR:
      return "Invalid repetition count";
    case REG_ERANGE:
      return "Invalid range end";
    case REG_ESPACE:
      return "Out of memory";
    case REG_BADRPT:
      return "Invalid preceding expression";
    default:
      return "Unknown error";
    }
  }();

  size_t msg_len = msg.size() + 1; // include NUL

  if (errbuf_size > 0 && errbuf) {
    size_t copy_len = msg_len < errbuf_size ? msg_len : errbuf_size;
    inline_memcpy(errbuf, msg.data(), copy_len - 1);
    errbuf[copy_len - 1] = '\0';
  }
  // POSIX requires returning the size needed to hold the full NUL-terminated
  // string, even if it was truncated in the buffer.
  return msg_len;
}

} // namespace LIBC_NAMESPACE_DECL
