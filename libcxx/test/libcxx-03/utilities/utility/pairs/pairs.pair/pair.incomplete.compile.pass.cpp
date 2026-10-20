//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that instantiating pair doesn't look up type traits "too early", before
// the contained types have been completed.
//
// This is a regression test, to prevent a reoccurrance of the issue introduced
// in 5e1de27f680591a870d78e9952b23f76aed7f456.

#include <utility>
#include <vector>

struct Test {
  std::vector<std::pair<int, Test> > v;
};

std::pair<int, Test> p;
