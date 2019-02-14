//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=37804

#include <map>
std::map<int,int>::iterator it;
#include <set>
using std::set;
using std::multiset;

int main(int, char**) { return 0; }
