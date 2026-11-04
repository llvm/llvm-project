//===-- Implementation of feof for baremetal --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/feof.h"

#include "hdr/types/FILE.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, feof, (::FILE * stream)) {
  (void)stream;
  // TODO: Shall we have an embeddeding API for feof?
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
