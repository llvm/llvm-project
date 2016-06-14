//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

#include <tuple>
#include <string>
#include <memory>

#include <cassert>

int main()
{
    typedef std::unique_ptr<int> upint;
    std::tuple<upint> t(upint(new int(4)));
    upint p = std::get<upint>(t);
}
