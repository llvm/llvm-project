//===-- Implementation of siglongjmp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/siglongjmp.h"
#include "src/__support/common.h"
#include "src/setjmp/longjmp.h"

namespace LIBC_NAMESPACE_DECL {

// siglongjmp is the same as longjmp. The additional recovery work is done in
// the epilogue of the sigsetjmp function.
// TODO: move this inside the TU of longjmp and making it an alias after
//       sigsetjmp is implemented for all architectures.
LLVM_LIBC_FUNCTION(void, siglongjmp, (jmp_buf buf, int val)) {
  return LIBC_NAMESPACE::longjmp(buf, val);
}

} // namespace LIBC_NAMESPACE_DECL
