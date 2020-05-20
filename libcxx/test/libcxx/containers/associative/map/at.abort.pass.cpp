//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// mapped_type& at(const key_type& k);

// Make sure we abort() when exceptions are disabled and we fetch a key that
// is not in the map.

// REQUIRES: no-exceptions

#include <csignal>
#include <cstdlib>
#include <map>

#include "test_macros.h"


void exit_success(int) {
    std::_Exit(EXIT_SUCCESS);
}

int main(int, char**) {
    std::signal(SIGABRT, exit_success);
    std::map<int, int> map;
    map.at(1);
    return EXIT_FAILURE;
}
