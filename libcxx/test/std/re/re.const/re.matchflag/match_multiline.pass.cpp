// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <regex>

// multiline:
//     Specifies that ^ shall match the beginning of a line and $ shall match
//     the end of a line, if the ECMAScript engine is selected.

#include <regex>
#include <cassert>
#include "test_macros.h"

static void search(const char* pat, std::regex_constants::syntax_option_type f,
                   const char* target, bool expected)
{
    std::regex re(pat, f);
    std::cmatch m;
    assert(std::regex_search(target, m, re) == expected);

    if(expected) {
        assert(m.size() == 1);
        assert(m.length(0) == 3);
        assert(m.str(0) == "foo");
    }
    else
    {
        assert(m.size() == 0);
    }
}

int main(int, char**)
{
    using std::regex_constants::ECMAScript;
    using std::regex_constants::basic;
    using std::regex_constants::extended;
    using std::regex_constants::awk;
    using std::regex_constants::grep;
    using std::regex_constants::egrep;
    using std::regex_constants::multiline;

    {
        const char* pat = "^foo";
        const char* target = "foo";

        search(pat, ECMAScript, target, true);
        search(pat, basic, target, true);
        search(pat, extended, target, true);
        search(pat, awk, target, true);
        search(pat, grep, target, true);
        search(pat, egrep, target, true);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, true);
        search(pat, extended | multiline, target, true);
        search(pat, awk | multiline, target, true);
        search(pat, grep | multiline, target, true);
        search(pat, egrep | multiline, target, true);
    }
    {
        const char* pat = "^foo";
        const char* target = "\nfoo";

        search(pat, ECMAScript, target, false);
        search(pat, basic, target, false);
        search(pat, extended, target, false);
        search(pat, awk, target, false);
        search(pat, grep, target, false);
        search(pat, egrep, target, false);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, false);
        search(pat, extended | multiline, target, false);
        search(pat, awk | multiline, target, false);
        search(pat, grep | multiline, target, false);
        search(pat, egrep | multiline, target, false);
    }
    {
        const char* pat = "^foo";
        const char* target = "bar\nfoo";

        search(pat, ECMAScript, target, false);
        search(pat, basic, target, false);
        search(pat, extended, target, false);
        search(pat, awk, target, false);
        search(pat, grep, target, false);
        search(pat, egrep, target, false);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, false);
        search(pat, extended | multiline, target, false);
        search(pat, awk | multiline, target, false);
        search(pat, grep | multiline, target, false);
        search(pat, egrep | multiline, target, false);
    }

    {
        const char* pat = "foo$";
        const char* target = "foo";

        search(pat, ECMAScript, target, true);
        search(pat, basic, target, true);
        search(pat, extended, target, true);
        search(pat, awk, target, true);
        search(pat, grep, target, true);
        search(pat, egrep, target, true);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, true);
        search(pat, extended | multiline, target, true);
        search(pat, awk | multiline, target, true);
        search(pat, grep | multiline, target, true);
        search(pat, egrep | multiline, target, true);
    }
    {
        const char* pat = "foo$";
        const char* target = "foo\n";

        search(pat, ECMAScript, target, false);
        search(pat, basic, target, false);
        search(pat, extended, target, false);
        search(pat, awk, target, false);
        search(pat, grep, target, false);
        search(pat, egrep, target, false);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, false);
        search(pat, extended | multiline, target, false);
        search(pat, awk | multiline, target, false);
        search(pat, grep | multiline, target, false);
        search(pat, egrep | multiline, target, false);
    }
    {
        const char* pat = "foo$";
        const char* target = "foo\nbar";

        search(pat, ECMAScript, target, false);
        search(pat, basic, target, false);
        search(pat, extended, target, false);
        search(pat, awk, target, false);
        search(pat, grep, target, false);
        search(pat, egrep, target, false);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, false);
        search(pat, extended | multiline, target, false);
        search(pat, awk | multiline, target, false);
        search(pat, grep | multiline, target, false);
        search(pat, egrep | multiline, target, false);
    }


    {
        const char* pat = "^foo";
        const char* target = "foo";

        search(pat, ECMAScript, target, true);
        search(pat, basic, target, true);
        search(pat, extended, target, true);
        search(pat, awk, target, true);
        search(pat, grep, target, true);
        search(pat, egrep, target, true);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, true);
        search(pat, extended | multiline, target, true);
        search(pat, awk | multiline, target, true);
        search(pat, grep | multiline, target, true);
        search(pat, egrep | multiline, target, true);
    }
    {
        const char* pat = "^foo";
        const char* target = "\rfoo";

        search(pat, ECMAScript, target, false);
        search(pat, basic, target, false);
        search(pat, extended, target, false);
        search(pat, awk, target, false);
        search(pat, grep, target, false);
        search(pat, egrep, target, false);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, false);
        search(pat, extended | multiline, target, false);
        search(pat, awk | multiline, target, false);
        search(pat, grep | multiline, target, false);
        search(pat, egrep | multiline, target, false);
    }
    {
        const char* pat = "^foo";
        const char* target = "bar\rfoo";

        search(pat, ECMAScript, target, false);
        search(pat, basic, target, false);
        search(pat, extended, target, false);
        search(pat, awk, target, false);
        search(pat, grep, target, false);
        search(pat, egrep, target, false);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, false);
        search(pat, extended | multiline, target, false);
        search(pat, awk | multiline, target, false);
        search(pat, grep | multiline, target, false);
        search(pat, egrep | multiline, target, false);
    }

    {
        const char* pat = "foo$";
        const char* target = "foo";

        search(pat, ECMAScript, target, true);
        search(pat, basic, target, true);
        search(pat, extended, target, true);
        search(pat, awk, target, true);
        search(pat, grep, target, true);
        search(pat, egrep, target, true);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, true);
        search(pat, extended | multiline, target, true);
        search(pat, awk | multiline, target, true);
        search(pat, grep | multiline, target, true);
        search(pat, egrep | multiline, target, true);
    }
    {
        const char* pat = "foo$";
        const char* target = "foo\r";

        search(pat, ECMAScript, target, false);
        search(pat, basic, target, false);
        search(pat, extended, target, false);
        search(pat, awk, target, false);
        search(pat, grep, target, false);
        search(pat, egrep, target, false);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, false);
        search(pat, extended | multiline, target, false);
        search(pat, awk | multiline, target, false);
        search(pat, grep | multiline, target, false);
        search(pat, egrep | multiline, target, false);
    }
    {
        const char* pat = "foo$";
        const char* target = "foo\rbar";

        search(pat, ECMAScript, target, false);
        search(pat, basic, target, false);
        search(pat, extended, target, false);
        search(pat, awk, target, false);
        search(pat, grep, target, false);
        search(pat, egrep, target, false);

        search(pat, ECMAScript | multiline, target, true);
        search(pat, basic | multiline, target, false);
        search(pat, extended | multiline, target, false);
        search(pat, awk | multiline, target, false);
        search(pat, grep | multiline, target, false);
        search(pat, egrep | multiline, target, false);
    }

    return 0;
}
