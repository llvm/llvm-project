//===-- Linux implementation of getpagesize -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getpagesize.h"

#include "src/__support/OSUtil/linux/auxv.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getpagesize, ()) {
  cpp::optional<unsigned long> page_size =
      (LIBC_NAMESPACE::auxv::get(AT_PAGESZ));
  if (page_size)
    return static_cast<int>(*page_size);
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
