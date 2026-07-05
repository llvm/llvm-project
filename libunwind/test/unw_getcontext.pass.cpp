// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#undef NDEBUG
#include <assert.h>
#include <libunwind.h>

int main(int, char**) {
  unw_context_t context;
  int ret = unw_getcontext(&context);
  assert(ret == UNW_ESUCCESS);
  return 0;
}
