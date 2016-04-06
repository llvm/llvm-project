//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <ext/hash_map>

namespace __gnu_cxx {
template class hash_map<int, int>;
}

int main() {
  typedef __gnu_cxx::hash_map<int, int> Map;
  Map m;
  Map m2(m);
  ((void)m2);
}
