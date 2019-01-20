//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>
// UNSUPPORTED: c++98, c++03

// Make sure that we correctly match inverted character classes.

#include <cassert>
#include <regex>


int main() {
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
}
