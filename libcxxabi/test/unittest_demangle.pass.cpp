//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -Wno-unused-function

#include <cassert>
#include <utility>

#include "../src/demangle/ItaniumDemangle.h"

void testPODSmallVector() {
  { // {push/pop}_back
    itanium_demangle::PODSmallVector<int, 1> PSV;
    PSV.push_back(0);
    PSV.push_back(1);
    PSV.push_back(2);
    PSV.push_back(3);
    for (int i = 0; i < 4; ++i)
      assert(PSV[i] == i);
    PSV.pop_back();
    for (int i = 0; i < 3; ++i)
      assert(PSV[i] == i);
    PSV.pop_back();
    PSV.pop_back();
    assert(!PSV.empty() && PSV.size() == 1);
    PSV.pop_back();
    assert(PSV.empty() && PSV.size() == 0);
  }

  {
    itanium_demangle::PODSmallVector<int, 1> PSV1;
    PSV1.push_back(1);
    PSV1.push_back(2);
    PSV1.push_back(3);

    itanium_demangle::PODSmallVector<int, 1> PSV2;
    std::swap(PSV1, PSV2);
    assert(PSV1.size() == 0);
    assert(PSV2.size() == 3);
    int i = 1;
    for (int x : PSV2) {
      assert(x == i);
      ++i;
    }
    assert(i == 4);
    std::swap(PSV1, PSV2);
    assert(PSV1.size() == 3);
    assert(PSV2.size() == 0);
    i = 1;
    for (int x : PSV1) {
      assert(x == i);
      ++i;
    }
    assert(i == 4);
  }

  {
    itanium_demangle::PODSmallVector<int, 10> PSV1;
    itanium_demangle::PODSmallVector<int, 10> PSV2;
    PSV1.push_back(0);
    PSV1.push_back(1);
    PSV1.push_back(2);
    assert(PSV1.size() == 3);
    assert(PSV2.size() == 0);
    std::swap(PSV1, PSV2);
    assert(PSV1.size() == 0);
    assert(PSV2.size() == 3);
    int i = 0;
    for (int x : PSV2) {
      assert(x == i);
      ++i;
    }
    for (int x : PSV1) {
      assert(false);
      (void)x;
    }
  }
}

int main(int, char**) {
  testPODSmallVector();
  return 0;
}
