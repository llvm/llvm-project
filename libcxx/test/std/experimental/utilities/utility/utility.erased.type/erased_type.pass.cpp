//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <experimental/utility>

#include <experimental/utility>

int main(int, char**)
{
  std::experimental::erased_type e;
  ((void)e);

  return 0;
}
