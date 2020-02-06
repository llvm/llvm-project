//===-- utilities.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/definitions.h"

namespace gwp_asan {
// Checks that `Condition` is true, otherwise fails in a platform-specific way
// with `Message`.
void Check(bool Condition, const char *Message);
} // namespace gwp_asan
