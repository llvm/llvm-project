//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// explicit basic_string(basic_string_view<CharT, traits> sv, const Allocator& a = Allocator());

#include <string>
#include <string_view>

void foo ( const string &s ) {}

int main()
{
    std::string_view sv = "ABCDE";
    foo(sv);    // requires implicit conversion from string_view to string
}
