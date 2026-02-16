//===-- GPU Implementation of fclose --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"

#include "hdr/stdio_macros.h"
#include "hdr/types/FILE.h"
#include "src/__support/common.h"
#include "src/stdio/gpu/file.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fclose, (::FILE * stream)) {
  return rpc::dispatch<LIBC_CLOSE_FILE>(rpc::client, fclose, stream);
}

} // namespace LIBC_NAMESPACE_DECL
