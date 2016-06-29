//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstreambuf

// int overflow(int c);

#include <iostream>
#include <string>
#include <strstream>

int main(int, char const **argv) {
  std::ostrstream oss;
  std::string s;

  for (int i = 0; i < 4096; ++i)
    s.push_back((i % 16) + 'a');

  oss << s << std::ends;
  std::cout << oss.str();
  oss.freeze(false);

  return 0;
}
