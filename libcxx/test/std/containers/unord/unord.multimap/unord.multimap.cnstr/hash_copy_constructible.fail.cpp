//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// The test requires access control SFINAE.

// GCC 5 does not evaluate static assertions dependent on a template parameter.
// UNSUPPORTED: gcc-5

// <unordered_map>

// Check that std::unordered_multimap fails to instantiate if the hash function is
// not copy-constructible. This is mentioned in LWG issue 2436

#include <unordered_map>

template <class T>
struct Hash {
    std::size_t operator () (const T& lhs) const { return 0; }

    Hash () {}
private:
    Hash (const Hash &); // declared but not defined
    };


int main(int, char**) {
    std::unordered_multimap<int, int, Hash<int> > m;

  return 0;
}
