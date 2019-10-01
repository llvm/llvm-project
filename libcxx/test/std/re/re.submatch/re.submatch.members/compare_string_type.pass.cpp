//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class BidirectionalIterator> class sub_match;

// int compare(const string_type& s) const;

#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
    {
        typedef char CharT;
        typedef std::sub_match<const CharT*> SM;
        typedef SM::string_type string;
        SM sm = SM();
        assert(sm.compare(string()) == 0);
        const CharT s[] = {'1', '2', '3', 0};
        sm.first = s;
        sm.second = s + 3;
        sm.matched = true;
        assert(sm.compare(string()) > 0);
        assert(sm.compare(string("123")) == 0);
    }
    {
        typedef wchar_t CharT;
        typedef std::sub_match<const CharT*> SM;
        typedef SM::string_type string;
        SM sm = SM();
        assert(sm.compare(string()) == 0);
        const CharT s[] = {'1', '2', '3', 0};
        sm.first = s;
        sm.second = s + 3;
        sm.matched = true;
        assert(sm.compare(string()) > 0);
        assert(sm.compare(string(L"123")) == 0);
    }

  return 0;
}
