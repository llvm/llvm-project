// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// namespace regex_constants
// {
//
// emum syntax_option_type  // bitmask type
// {
//     icase      = unspecified,
//     nosubs     = unspecified,
//     optimize   = unspecified,
//     collate    = unspecified,
//     ECMAScript = unspecified,
//     basic      = unspecified,
//     extended   = unspecified,
//     awk        = unspecified,
//     grep       = unspecified,
//     egrep      = unspecified,
//     multiline  = unspecified
// };
//
// }

#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
    assert(std::regex_constants::icase != 0);
    assert(std::regex_constants::nosubs != 0);
    assert(std::regex_constants::optimize != 0);
    assert(std::regex_constants::collate != 0);
#ifdef _LIBCPP_ABI_REGEX_CONSTANTS_NONZERO  // https://bugs.llvm.org/show_bug.cgi?id=35967
    assert(std::regex_constants::ECMAScript != 0);
#else
    assert(std::regex_constants::ECMAScript == 0);
#endif
    assert(std::regex_constants::basic != 0);
    assert(std::regex_constants::extended != 0);
    assert(std::regex_constants::awk != 0);
    assert(std::regex_constants::grep != 0);
    assert(std::regex_constants::egrep != 0);
    assert(std::regex_constants::multiline != 0);

    assert((std::regex_constants::icase & std::regex_constants::nosubs) == 0);
    assert((std::regex_constants::icase & std::regex_constants::optimize) == 0);
    assert((std::regex_constants::icase & std::regex_constants::collate) == 0);
    assert((std::regex_constants::icase & std::regex_constants::ECMAScript) == 0);
    assert((std::regex_constants::icase & std::regex_constants::basic) == 0);
    assert((std::regex_constants::icase & std::regex_constants::extended) == 0);
    assert((std::regex_constants::icase & std::regex_constants::awk) == 0);
    assert((std::regex_constants::icase & std::regex_constants::grep) == 0);
    assert((std::regex_constants::icase & std::regex_constants::egrep) == 0);
    assert((std::regex_constants::icase & std::regex_constants::multiline) == 0);

    assert((std::regex_constants::nosubs & std::regex_constants::optimize) == 0);
    assert((std::regex_constants::nosubs & std::regex_constants::collate) == 0);
    assert((std::regex_constants::nosubs & std::regex_constants::ECMAScript) == 0);
    assert((std::regex_constants::nosubs & std::regex_constants::basic) == 0);
    assert((std::regex_constants::nosubs & std::regex_constants::extended) == 0);
    assert((std::regex_constants::nosubs & std::regex_constants::awk) == 0);
    assert((std::regex_constants::nosubs & std::regex_constants::grep) == 0);
    assert((std::regex_constants::nosubs & std::regex_constants::egrep) == 0);
    assert((std::regex_constants::nosubs & std::regex_constants::multiline) == 0);

    assert((std::regex_constants::optimize & std::regex_constants::collate) == 0);
    assert((std::regex_constants::optimize & std::regex_constants::ECMAScript) == 0);
    assert((std::regex_constants::optimize & std::regex_constants::basic) == 0);
    assert((std::regex_constants::optimize & std::regex_constants::extended) == 0);
    assert((std::regex_constants::optimize & std::regex_constants::awk) == 0);
    assert((std::regex_constants::optimize & std::regex_constants::grep) == 0);
    assert((std::regex_constants::optimize & std::regex_constants::egrep) == 0);
    assert((std::regex_constants::optimize & std::regex_constants::multiline) == 0);

    assert((std::regex_constants::collate & std::regex_constants::ECMAScript) == 0);
    assert((std::regex_constants::collate & std::regex_constants::basic) == 0);
    assert((std::regex_constants::collate & std::regex_constants::extended) == 0);
    assert((std::regex_constants::collate & std::regex_constants::awk) == 0);
    assert((std::regex_constants::collate & std::regex_constants::grep) == 0);
    assert((std::regex_constants::collate & std::regex_constants::egrep) == 0);
    assert((std::regex_constants::collate & std::regex_constants::multiline) == 0);

    assert((std::regex_constants::ECMAScript & std::regex_constants::basic) == 0);
    assert((std::regex_constants::ECMAScript & std::regex_constants::extended) == 0);
    assert((std::regex_constants::ECMAScript & std::regex_constants::awk) == 0);
    assert((std::regex_constants::ECMAScript & std::regex_constants::grep) == 0);
    assert((std::regex_constants::ECMAScript & std::regex_constants::egrep) == 0);
    assert((std::regex_constants::ECMAScript & std::regex_constants::multiline) == 0);

    assert((std::regex_constants::basic & std::regex_constants::extended) == 0);
    assert((std::regex_constants::basic & std::regex_constants::awk) == 0);
    assert((std::regex_constants::basic & std::regex_constants::grep) == 0);
    assert((std::regex_constants::basic & std::regex_constants::egrep) == 0);
    assert((std::regex_constants::basic & std::regex_constants::multiline) == 0);

    assert((std::regex_constants::extended & std::regex_constants::awk) == 0);
    assert((std::regex_constants::extended & std::regex_constants::grep) == 0);
    assert((std::regex_constants::extended & std::regex_constants::egrep) == 0);
    assert((std::regex_constants::extended & std::regex_constants::multiline) == 0);

    assert((std::regex_constants::awk & std::regex_constants::grep) == 0);
    assert((std::regex_constants::awk & std::regex_constants::egrep) == 0);
    assert((std::regex_constants::awk & std::regex_constants::multiline) == 0);

    assert((std::regex_constants::grep & std::regex_constants::egrep) == 0);
    assert((std::regex_constants::grep & std::regex_constants::multiline) == 0);

    assert((std::regex_constants::egrep & std::regex_constants::multiline) == 0);

    assert((std::regex_constants::icase | std::regex_constants::nosubs) != 0);
    assert((std::regex_constants::icase ^ std::regex_constants::nosubs) != 0);

    std::regex_constants::syntax_option_type e1 = std::regex_constants::icase;
    std::regex_constants::syntax_option_type e2 = std::regex_constants::nosubs;
    e1 = ~e1;
    e1 = e1 & e2;
    e1 = e1 | e2;
    e1 = e1 ^ e2;
    e1 &= e2;
    e1 |= e2;
    e1 ^= e2;

  return 0;
}
