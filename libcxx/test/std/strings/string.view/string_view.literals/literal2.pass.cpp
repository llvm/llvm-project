// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// UNSUPPORTED: clang-3.3, clang-3.4, clang-3.5, clang-3.6, clang-3.7, clang-3.8, clang-3.9
// UNSUPPORTED: apple-clang-6, apple-clang-7, apple-clang-8
// Note: libc++ supports string_view before C++17, but literals were introduced in C++14

#include <string_view>
#include <cassert>

int main(int, char**)
{
    using namespace std::literals::string_view_literals;

    std::string_view foo  =   ""sv;
    assert(foo.length() == 0);

  return 0;
}
