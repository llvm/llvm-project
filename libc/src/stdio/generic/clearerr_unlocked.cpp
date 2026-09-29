//===-- Implementation of clearerr_unlocked -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/clearerr_unlocked.h"
#include "src/__support/File/file.h"
#include "src/__support/macros/config.h"

#include "hdr/types/FILE.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, clearerr_unlocked, (::FILE * stream)) {
  reinterpret_cast<LIBC_NAMESPACE::File *>(stream)->clearerr_unlocked();
}

} // namespace LIBC_NAMESPACE_DECL
