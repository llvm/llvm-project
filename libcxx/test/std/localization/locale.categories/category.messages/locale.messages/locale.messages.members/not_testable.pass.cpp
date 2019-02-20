//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class messages<charT>

// catalog open(const basic_string<char>& name, const locale&) const;

#include <locale>
#include <cassert>

// As far as I can tell, the messages facet is untestable.  I have a best
// effort implementation in the hopes that in the future I will learn how
// to test it.

template <class CharT>
class F
    : public std::messages<CharT>
{
public:
    explicit F(std::size_t refs = 0)
        : std::messages<CharT>(refs) {}
};

int main(int, char**)
{

  return 0;
}
