//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <strstream>

// class strstreambuf

// int_type underflow();

#include <strstream>
#include <cassert>

#include "test_macros.h"

struct test
    : public std::strstreambuf
{
    typedef std::strstreambuf base;
    test(char* gnext_arg, std::streamsize n, char* pbeg_arg = 0)
        : base(gnext_arg, n, pbeg_arg) {}
    test(const char* gnext_arg, std::streamsize n)
        : base(gnext_arg, n) {}

    base::int_type underflow() {return base::underflow();}
};

int main(int, char**)
{
    {
        char buf[10] = "123";
        test sb(buf, 0, buf + 3);
        assert(sb.underflow() == '1');
        assert(sb.underflow() == '1');
        assert(sb.snextc() == '2');
        assert(sb.underflow() == '2');
        assert(sb.underflow() == '2');
        assert(sb.snextc() == '3');
        assert(sb.underflow() == '3');
        assert(sb.underflow() == '3');
        assert(sb.snextc() == EOF);
        assert(sb.underflow() == EOF);
        assert(sb.underflow() == EOF);
        sb.sputc('4');
        assert(sb.underflow() == '4');
        assert(sb.underflow() == '4');
    }

  return 0;
}
