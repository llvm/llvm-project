//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <streambuf>

// template <class charT, class traits = char_traits<charT> >
// class basic_streambuf;

// void pbump(int n);
//
// REQUIRES: long_tests

#include <sstream>
#include <cassert>
#include "test_macros.h"

struct SB : std::stringbuf
{
  SB() : std::stringbuf(std::ios::ate|std::ios::out) { }
  const char* pubpbase() const { return pbase(); }
  const char* pubpptr() const { return pptr(); }
};

int main()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
#endif
    	std::string str(2147483648, 'a');
		SB sb;
		sb.str(str);
		assert(sb.pubpbase() <= sb.pubpptr());
#ifndef TEST_HAS_NO_EXCEPTIONS
	}
	catch (const std::bad_alloc &) {}
#endif
}
