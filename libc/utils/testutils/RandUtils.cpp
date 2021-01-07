//===-- RandUtils.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RandUtils.h"

#include <cstdlib>

namespace __llvm_libc {
namespace testutils {

int rand() { return std::rand(); }

} // namespace testutils
} // namespace __llvm_libc
