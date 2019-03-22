//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT, class Traits, class Allocator>
//   bool operator()(const basic_string<charT,Traits,Allocator>& s1,
//                   const basic_string<charT,Traits,Allocator>& s2) const;

#include <locale>
#include <cassert>

int main(int, char**)
{
    {
        std::locale l;
        {
            std::string s2("aaaaaaA");
            std::string s3("BaaaaaA");
            assert(l(s3, s2));
        }
        {
            std::wstring s2(L"aaaaaaA");
            std::wstring s3(L"BaaaaaA");
            assert(l(s3, s2));
        }
    }

  return 0;
}
