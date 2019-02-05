//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// Check that std::set fails to instantiate if the comparison predicate is
// not copy-constructible. This is LWG issue 2436

#include <set>

template <class T>
struct Comp {
    bool operator () (const T& lhs, const T& rhs) const { return lhs < rhs; }

    Comp () {}
private:
    Comp (const Comp &); // declared but not defined
    };


int main(int, char**) {
    std::set<int, Comp<int> > m;

  return 0;
}
