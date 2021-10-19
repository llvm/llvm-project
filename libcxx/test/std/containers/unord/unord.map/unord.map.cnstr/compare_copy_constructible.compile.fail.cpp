//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// The test requires access control SFINAE.

// <unordered_map>

// Check that std::unordered_map fails to instantiate if the comparison predicate is
// not copy-constructible. This is LWG issue 2436

#include <unordered_map>

template <class T>
struct Comp {
    bool operator () (const T& lhs, const T& rhs) const { return lhs == rhs; }

    Comp () {}
private:
    Comp (const Comp &); // declared but not defined
    };


int main(int, char**) {
    std::unordered_map<int, int, std::hash<int>, Comp<int> > m;

  return 0;
}
