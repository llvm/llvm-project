//===-- Implementation of insque --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/insque.h"
#include "src/__support/common.h"
#include "src/__support/intrusive_list.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void, insque, (void *elem, void *prev)) {
  internal::IntrusiveList::insert(elem, prev);
}

} // namespace LIBC_NAMESPACE
