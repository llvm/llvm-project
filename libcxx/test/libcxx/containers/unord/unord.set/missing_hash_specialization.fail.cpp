//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// REQUIRES: diagnose-if-support
// UNSUPPORTED: c++98, c++03

// Libc++ only provides a defined primary template for std::hash in C++14 and
// newer.
// UNSUPPORTED: c++11

// <unordered_set>

// Test that we generate a reasonable diagnostic when the specified hash is
// not enabled.

#include <unordered_set>
#include <utility>

using VT = std::pair<int, int>;
using Set = std::unordered_set<VT>;

int main() {

  Set s; // expected-error@__hash_table:* {{the specified hash functor does not meet the requirements for an enabled hash}}

  // FIXME: It would be great to suppress the below diagnostic all together.
  //        but for now it's sufficient that it appears last. However there is
  //        currently no way to test the order diagnostics are issued.
  // expected-error@memory:* {{call to implicitly-deleted default constructor of 'std::__1::hash<std::__1::pair<int, int> >'}}
}
