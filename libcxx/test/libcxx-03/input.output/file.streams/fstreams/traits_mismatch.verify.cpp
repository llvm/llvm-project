//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_fstream

// The char type of the stream and the char_type of the traits have to match

// UNSUPPORTED: no-wide-characters

#include <fstream>

std::basic_fstream<char, std::char_traits<wchar_t> > f;
// expected-error-re@*:* {{static assertion failed{{.*}}traits_type::char_type must be the same type as CharT}}
// expected-error-re@*:* {{static assertion failed{{.*}}traits_type::char_type must be the same type as CharT}}

// expected-error@*:* 11 {{only virtual member functions can be marked 'override'}}

// FIXME: As of commit r324062 Clang incorrectly generates a diagnostic about mismatching
// exception specifications for types which are already invalid for one reason or another.
// For now we tolerate this diagnostic.
// expected-error@*:* 0-1 {{exception specification of overriding function is more lax than base version}}
