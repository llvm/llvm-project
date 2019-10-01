//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/functional>
//
//  has to include <functional>

#include <experimental/functional>

#include "test_macros.h"

int main(int, char**)
{
  std::function<int(int)> x;

  return 0;
}
