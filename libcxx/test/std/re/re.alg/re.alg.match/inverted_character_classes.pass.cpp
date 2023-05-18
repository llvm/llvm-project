//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>
// UNSUPPORTED: c++03

// Make sure that we correctly match inverted character classes.

#include <cassert>
#include <regex>

#include "test_macros.h"


int main(int, char**) {
    assert(std::regex_match("X", std::regex("[X]")));
    assert(std::regex_match("X", std::regex("[XY]")));
    assert(!std::regex_match("X", std::regex("[^X]")));
    assert(!std::regex_match("X", std::regex("[^XY]")));

    assert(std::regex_match("X", std::regex("[\\S]")));
    assert(!std::regex_match("X", std::regex("[^\\S]")));

    assert(!std::regex_match("X", std::regex("[\\s]")));
    assert(std::regex_match("X", std::regex("[^\\s]")));

    assert(std::regex_match("X", std::regex("[\\s\\S]")));
    assert(std::regex_match("X", std::regex("[^Y\\s]")));
    assert(!std::regex_match("X", std::regex("[^X\\s]")));

    assert(std::regex_match("X", std::regex("[\\w]")));
    assert(std::regex_match("_", std::regex("[\\w]")));
    assert(!std::regex_match("X", std::regex("[^\\w]")));
    assert(!std::regex_match("_", std::regex("[^\\w]")));

    assert(!std::regex_match("X", std::regex("[\\W]")));
    assert(!std::regex_match("_", std::regex("[\\W]")));
    assert(std::regex_match("X", std::regex("[^\\W]")));
    assert(std::regex_match("_", std::regex("[^\\W]")));

    // Those test cases are taken from PR40904
    assert(std::regex_match("abZcd", std::regex("^ab[\\d\\D]cd")));
    assert(std::regex_match("ab5cd", std::regex("^ab[\\d\\D]cd")));
    assert(std::regex_match("abZcd", std::regex("^ab[\\D]cd")));
    assert(std::regex_match("abZcd", std::regex("^ab\\Dcd")));
    assert(std::regex_match("ab5cd", std::regex("^ab[\\d]cd")));
    assert(std::regex_match("ab5cd", std::regex("^ab\\dcd")));
    assert(!std::regex_match("abZcd", std::regex("^ab\\dcd")));
    assert(!std::regex_match("ab5cd", std::regex("^ab\\Dcd")));

    assert(std::regex_match("_xyz_", std::regex("_(\\s|\\S)+_")));
    assert(std::regex_match("_xyz_", std::regex("_[\\s\\S]+_")));

    return 0;
}
