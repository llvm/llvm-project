//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test that cmath builds with -fdelayed-template-parsing

// REQUIRES: fdelayed-template-parsing

// RUN: %build -fdelayed-template-parsing
// RUN: %run

#include <cmath>
#include <cassert>

#include "test_macros.h"

int main() {
  assert(std::isfinite(1.0));
  assert(!std::isinf(1.0));
  assert(!std::isnan(1.0));
}

using namespace std;
