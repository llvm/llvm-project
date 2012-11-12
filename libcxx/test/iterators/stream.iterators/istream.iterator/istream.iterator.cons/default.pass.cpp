//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// class istream_iterator

// istream_iterator();

#include <iterator>
#include <cassert>

int main()
{
    std::istream_iterator<int> i;
    assert(i == std::istream_iterator<int>());
}
