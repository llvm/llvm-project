//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

//  template<class charT, class traits = char_traits<charT>,
//           class Allocator = allocator<charT>>
//    class basic_stringbuf;
//
// The char type of the stream and the char_type of the traits have to match

// UNSUPPORTED: no-wide-characters

#include <sstream>

int main(int, char**)
{
	std::basic_stringbuf<char, std::char_traits<wchar_t> > sb;
//  expected-error-re@streambuf:* {{static assertion failed{{.*}}traits_type::char_type must be the same type as CharT}}
//  expected-error-re@string:* {{static assertion failed{{.*}}traits_type::char_type must be the same type as CharT}}

  return 0;
}
