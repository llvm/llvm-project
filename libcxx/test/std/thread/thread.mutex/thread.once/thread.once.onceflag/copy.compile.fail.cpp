//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// <mutex>

// struct once_flag;

// once_flag(const once_flag&) = delete;

#include <mutex>

int main(int, char**)
{
    std::once_flag f;
    std::once_flag f2(f);

  return 0;
}
