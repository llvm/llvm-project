//===-- Implementation of fflush ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fflush.h"

#include "file.h"
#include "hdr/types/FILE.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fflush, (::FILE * stream)) {
  return rpc::dispatch<LIBC_FFLUSH>(
      rpc::client, fflush, reinterpret_cast<FILE *>(file::from_stream(stream)));
}

} // namespace LIBC_NAMESPACE_DECL
