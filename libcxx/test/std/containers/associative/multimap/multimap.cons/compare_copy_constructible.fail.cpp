//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// Check that std::multimap fails to instantiate if the comparison predicate is
// not copy-constructible. This is LWG issue 2436

#include <map>

template <class T>
struct Comp {
    bool operator () (const T& lhs, const T& rhs) const { return lhs < rhs; }

    Comp () {}
private:
    Comp (const Comp &); // declared but not defined
    };


int main() {
    std::multimap<int, int, Comp<int> > m;
}
