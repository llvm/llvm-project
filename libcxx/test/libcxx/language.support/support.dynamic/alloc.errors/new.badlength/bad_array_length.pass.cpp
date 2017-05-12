//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// XFAIL: availability

// XFAIL: availability=macosx10.12
// XFAIL: availability=macosx10.11
// XFAIL: availability=macosx10.10
// XFAIL: availability=macosx10.9
// XFAIL: availability=macosx10.7
// XFAIL: availability=macosx10.8

// test bad_array_length

#include <new>
#include <type_traits>
#include <cassert>

int main()
{
    static_assert((std::is_base_of<std::bad_alloc, std::bad_array_length>::value),
                  "std::is_base_of<std::bad_alloc, std::bad_array_length>::value");
    static_assert(std::is_polymorphic<std::bad_array_length>::value,
                 "std::is_polymorphic<std::bad_array_length>::value");
    std::bad_array_length b;
    std::bad_array_length b2 = b;
    b2 = b;
    const char* w = b2.what();
    assert(w);
}
