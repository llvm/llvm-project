//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Tests workaround for  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64816.

#include <string>

void f(const std::string &s) { (void)s.begin(); }

#include <vector>

void AppendTo(const std::vector<char> &v) { (void)v.begin(); }

int main() {}
