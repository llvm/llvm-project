//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_multimap

// float max_load_factor() const;
// void max_load_factor(float mlf);

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <unordered_map>
#include <string>

#include "check_assertion.h"

int main(int, char**) {
    typedef std::unordered_multimap<int, std::string> C;
    C c;
    TEST_LIBCPP_ASSERT_FAILURE(c.max_load_factor(0), "unordered container::max_load_factor(lf) called with lf <= 0");

    return 0;
}
