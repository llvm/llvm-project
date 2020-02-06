//===-- utilities_posix.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/utilities.h"

#ifdef ANDROID
#include <android/set_abort_message.h>
#include <stdlib.h>
#else // ANDROID
#include <stdio.h>
#endif

namespace gwp_asan {

#ifdef ANDROID
void Check(bool Condition, const char *Message) {
  if (Condition)
    return;
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
