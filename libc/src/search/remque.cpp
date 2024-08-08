//===-- Implementation of remque --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/remque.h"
#include "src/__support/common.h"
#include "src/__support/intrusive_list.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, remque, (void *elem)) {
  internal::IntrusiveList::remove(elem);
}

} // namespace LIBC_NAMESPACE_DECL
