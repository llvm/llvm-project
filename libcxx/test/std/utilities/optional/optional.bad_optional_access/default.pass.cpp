//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <optional>

// class bad_optional_access is default constructible

#include <optional>
#include <type_traits>

int main()
{
    using std::bad_optional_access;
    bad_optional_access ex;
}
