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
// struct __is_inplace_tag;

#include <utility>
#include <cassert>

template <bool Expect, class RefFn, class Fn = std::remove_reference_t<RefFn>>
void do_test() {
    static_assert(std::__is_inplace_tag<RefFn>::value == Expect, "");
    static_assert(std::__is_inplace_tag<Fn>::value == Expect, "");
    static_assert(std::__is_inplace_tag<std::decay_t<RefFn>>::value == Expect, "");
    static_assert(std::__is_inplace_tag<Fn*>::value == Expect, "");
}

int main() {
    do_test<true, std::in_place_t>();
    do_test<true, std::in_place_type_t<int>>();
    do_test<true, std::in_place_index_t<42>>();
    do_test<false, std::in_place_tag>();
    do_test<false, void>();
    do_test<false, void*>();
    do_test<false, std::in_place_tag(&)(...)>();
}