//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getprotobyname_r.
///
//===----------------------------------------------------------------------===//

#include "src/netdb/getprotobyname_r.h"

#include "hdr/types/size_t.h"
#include "hdr/types/struct_protoent.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/null_check.h"
#include "src/netdb/getproto_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getprotobyname_r,
                   (const char *__restrict name,
                    struct protoent *__restrict result_buf,
                    char *__restrict buf, size_t buflen,
                    struct protoent **__restrict result)) {
  LIBC_CRASH_ON_NULLPTR(name);
  LIBC_CRASH_ON_NULLPTR(result_buf);
  if (buflen > 0)
    LIBC_CRASH_ON_NULLPTR(buf);
  LIBC_CRASH_ON_NULLPTR(result);

  return netdb::lookup_proto_by_name(name, result_buf, {buf, buflen}, result,
                                     netdb::get_protocols_path());
}

} // namespace LIBC_NAMESPACE_DECL
