//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// void clear(iostate state);

// Make sure that we abort() when exceptions are disabled and the exception
// flag is set for the iostate we pass to clear().

// REQUIRES: no-exceptions

#include <csignal>
#include <cstdlib>
#include <ios>
#include <streambuf>

#include "test_macros.h"


void exit_success(int) {
    std::_Exit(EXIT_SUCCESS);
}

struct testbuf : public std::streambuf {};

int main(int, char**) {
    std::signal(SIGABRT, exit_success);

    testbuf buf;
    std::ios ios(&buf);
    ios.exceptions(std::ios::badbit);
    ios.clear(std::ios::badbit);

    return EXIT_FAILURE;
}
