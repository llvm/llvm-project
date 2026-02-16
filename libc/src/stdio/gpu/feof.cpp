//===-- Implementation of feof --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/feof.h"

#include "file.h"
#include "hdr/types/FILE.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, feof, (::FILE * stream)) {
  return rpc::dispatch<LIBC_FEOF>(
      rpc::client, feof, reinterpret_cast<FILE *>(file::from_stream(stream)));
}

} // namespace LIBC_NAMESPACE_DECL
