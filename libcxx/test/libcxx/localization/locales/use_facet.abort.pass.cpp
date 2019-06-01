//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class Facet> const Facet& use_facet(const locale& loc);

// REQUIRES: libcpp-no-exceptions

// Make sure we abort() when we pass a facet not associated to the locale to
// use_facet() and exceptions are disabled.

#include <csignal>
#include <cstdlib>
#include <locale>

#include "test_macros.h"


struct my_facet : public std::locale::facet {
    static std::locale::id id;
};

std::locale::id my_facet::id;

void exit_success(int) {
    std::_Exit(EXIT_SUCCESS);
}

int main(int, char**) {
    std::signal(SIGABRT, exit_success);
    std::use_facet<my_facet>(std::locale());
    return EXIT_FAILURE;
}
