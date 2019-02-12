//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// class unordered_map

// mapped_type& at(const key_type& k);

// Make sure we abort() when exceptions are disabled and we fetch a key that
// is not in the map.

// REQUIRES: libcpp-no-exceptions
// UNSUPPORTED: c++98, c++03

#include <csignal>
#include <cstdlib>
#include <unordered_map>


int main(int, char**) {
    std::signal(SIGABRT, [](int) { std::_Exit(EXIT_SUCCESS); });
    std::unordered_map<int, int> map;
    map.at(1);
    return EXIT_FAILURE;
}
