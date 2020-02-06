//===-- utilities_posix.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/definitions.h"
#include "gwp_asan/utilities.h"

#ifdef ANDROID
#include <stdlib.h>
extern "C" GWP_ASAN_WEAK void android_set_abort_message(const char *);
#else // ANDROID
#include <stdio.h>
#endif

namespace gwp_asan {

#ifdef ANDROID
void Check(bool Condition, const char *Message) {
  if (Condition)
    return;
  if (&android_set_abort_message != nullptr)
    android_set_abort_message(Message);
  abort();
}
#else  // ANDROID
void Check(bool Condition, const char *Message) {
  if (Condition)
    return;
  fprintf(stderr, "%s", Message);
  __builtin_trap();
}
#endif // ANDROID

} // namespace gwp_asan
