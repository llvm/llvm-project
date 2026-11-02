//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of gai_strerror.
///
//===----------------------------------------------------------------------===//

#include "src/netdb/gai_strerror.h"
#include "hdr/netdb_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(const char *, gai_strerror, (int errcode)) {
  switch (errcode) {
  case EAI_AGAIN:
    return "Name could not be resolved at this time";
  case EAI_BADFLAGS:
    return "Flags had an invalid value";
  case EAI_FAIL:
    return "Non-recoverable error occurred";
  case EAI_FAMILY:
    return "Address family not recognized";
  case EAI_MEMORY:
    return "Memory allocation failure";
  case EAI_NONAME:
    return "Name does not resolve for the supplied parameters";
  case EAI_OVERFLOW:
    return "Argument buffer overflowed";
  case EAI_SERVICE:
    return "Service not recognized for specified socket type";
  case EAI_SOCKTYPE:
    return "Intended socket type not recognized";
  case EAI_SYSTEM:
    return "System error";
  default:
    return "Unknown error";
  }
}

} // namespace LIBC_NAMESPACE_DECL
