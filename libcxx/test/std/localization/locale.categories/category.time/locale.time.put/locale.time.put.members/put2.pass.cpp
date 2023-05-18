//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class time_put<charT, OutputIterator>

// iter_type put(iter_type s, ios_base& str, char_type fill, const tm* t,
//               char format, char modifier = 0) const;

#include <locale>
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

typedef std::time_put<char, cpp17_output_iterator<char*> > F;

class my_facet
    : public F
{
public:
    explicit my_facet(std::size_t refs = 0)
        : F(refs) {}
};

int main(int, char**)
{
    const my_facet f(1);
    char str[200];
    tm t = {};
    t.tm_sec = 6;
    t.tm_min = 3;
    t.tm_hour = 13;
    t.tm_mday = 2;
    t.tm_mon = 4;
    t.tm_year = 109;
    t.tm_wday = 6;
    t.tm_yday = 121;
    t.tm_isdst = 1;
    std::ios ios(0);
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'A');
        std::string ex(str, base(iter));
        assert(ex == "Saturday");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'a');
        std::string ex(str, base(iter));
        assert(ex == "Sat");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'B');
        std::string ex(str, base(iter));
        assert(ex == "May");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'b');
        std::string ex(str, base(iter));
        assert(ex == "May");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'C');
        std::string ex(str, base(iter));
        assert(ex == "20");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'c');
        std::string ex(str, base(iter));
        assert(ex == "Sat May  2 13:03:06 2009");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'D');
        std::string ex(str, base(iter));
        assert(ex == "05/02/09");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'd');
        std::string ex(str, base(iter));
        assert(ex == "02");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'c', 'E');
        std::string ex(str, base(iter));
        assert(ex == "Sat May  2 13:03:06 2009");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'C', 'E');
        std::string ex(str, base(iter));
        assert(ex == "20");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'x', 'E');
        std::string ex(str, base(iter));
        assert(ex == "05/02/09");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'X', 'E');
        std::string ex(str, base(iter));
        assert(ex == "13:03:06");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'y', 'E');
        std::string ex(str, base(iter));
        assert(ex == "09");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'Y', 'E');
        std::string ex(str, base(iter));
        assert(ex == "2009");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'd', 'O');
        std::string ex(str, base(iter));
        assert(ex == "02");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'e', 'O');
        std::string ex(str, base(iter));
        assert(ex == " 2");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'H', 'O');
        std::string ex(str, base(iter));
        assert(ex == "13");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'I', 'O');
        std::string ex(str, base(iter));
        assert(ex == "01");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'm', 'O');
        std::string ex(str, base(iter));
        assert(ex == "05");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'M', 'O');
        std::string ex(str, base(iter));
        assert(ex == "03");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'S', 'O');
        std::string ex(str, base(iter));
        assert(ex == "06");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'u', 'O');
        std::string ex(str, base(iter));
        assert(ex == "6");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'U', 'O');
        std::string ex(str, base(iter));
        assert(ex == "17");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'V', 'O');
        std::string ex(str, base(iter));
        assert(ex == "18");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'w', 'O');
        std::string ex(str, base(iter));
        assert(ex == "6");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'W', 'O');
        std::string ex(str, base(iter));
        assert(ex == "17");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'y', 'O');
        std::string ex(str, base(iter));
        assert(ex == "09");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'e');
        std::string ex(str, base(iter));
        assert(ex == " 2");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'F');
        std::string ex(str, base(iter));
        assert(ex == "2009-05-02");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'G');
        std::string ex(str, base(iter));
        assert(ex == "2009");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'g');
        std::string ex(str, base(iter));
        assert(ex == "09");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'H');
        std::string ex(str, base(iter));
        assert(ex == "13");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'h');
        std::string ex(str, base(iter));
        assert(ex == "May");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'I');
        std::string ex(str, base(iter));
        assert(ex == "01");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'j');
        std::string ex(str, base(iter));
        assert(ex == "122");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'M');
        std::string ex(str, base(iter));
        assert(ex == "03");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'm');
        std::string ex(str, base(iter));
        assert(ex == "05");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'n');
        std::string ex(str, base(iter));
        assert(ex == "\n");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'p');
        std::string ex(str, base(iter));
        assert(ex == "PM");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'R');
        std::string ex(str, base(iter));
        assert(ex == "13:03");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'r');
        std::string ex(str, base(iter));
        assert(ex == "01:03:06 PM");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'S');
        std::string ex(str, base(iter));
        assert(ex == "06");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'T');
        std::string ex(str, base(iter));
        assert(ex == "13:03:06");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 't');
        std::string ex(str, base(iter));
        assert(ex == "\t");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'U');
        std::string ex(str, base(iter));
        assert(ex == "17");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'u');
        std::string ex(str, base(iter));
        assert(ex == "6");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'V');
        std::string ex(str, base(iter));
        assert(ex == "18");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'W');
        std::string ex(str, base(iter));
        assert(ex == "17");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'w');
        std::string ex(str, base(iter));
        assert(ex == "6");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'X');
        std::string ex(str, base(iter));
        assert(ex == "13:03:06");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'x');
        std::string ex(str, base(iter));
        assert(ex == "05/02/09");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'Y');
        std::string ex(str, base(iter));
        assert(ex == "2009");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'y');
        std::string ex(str, base(iter));
        assert(ex == "09");
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'Z');
        std::string ex(str, base(iter));
//        assert(ex == "EDT");  depends on time zone
    }
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, 'z');
        std::string ex(str, base(iter));
//        assert(ex == "-0400");  depends on time zone
    }
#ifndef _WIN32
    // The Windows strftime() doesn't support the "%+" format. Depending on CRT
    // configuration of the invalid parameter handler, this can abort the process.
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, '+');
        std::string ex(str, base(iter));
//        assert(ex == "Sat May  2 13:03:06 EDT 2009");  depends on time zone
    }
#endif
    {
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', &t, '%');
        std::string ex(str, base(iter));
        assert(ex == "%");
    }

  return 0;
}
