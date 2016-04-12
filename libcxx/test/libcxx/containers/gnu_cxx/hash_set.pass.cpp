//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <ext/hash_set>

namespace __gnu_cxx {
template class hash_set<int>;
}

int main() {
  typedef __gnu_cxx::hash_set<int> Set;
  Set s;
  Set s2(s);
  ((void)s2);
}
