//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <utility>

// template <class Tp>
// struct __is_inplace_type;

#include <utility>
#include <cassert>

int main() {
    static_assert(std::__is_inplace_type<std::in_place_t>::value, "");
    static_assert(std::__is_inplace_type<std::in_place_type_t<int>>::value, "");
    static_assert(std::__is_inplace_type<std::in_place_index_t<static_cast<size_t>(-1)>>::value, "");
    static_assert(!std::__is_inplace_type<std::in_place_tag>::value, "");
    static_assert(!std::__is_inplace_type<void*>::value, "");
    static_assert(!std::__is_inplace_type<std::in_place_tag(&)(...)>::value, "");
}