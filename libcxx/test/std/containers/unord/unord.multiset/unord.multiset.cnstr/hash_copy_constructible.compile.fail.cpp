//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


// <unordered_set>

// Check that std::unordered_multiset fails to instantiate if the hash function is
// not copy-constructible. This is mentioned in LWG issue 2436

#include <unordered_set>

template <class T>
struct Hash {
    std::size_t operator () (const T& lhs) const { return 0; }

    Hash () {}
private:
    Hash (const Hash &); // declared but not defined
    };


int main(int, char**) {
    std::unordered_multiset<int, Hash<int> > m;

  return 0;
}
