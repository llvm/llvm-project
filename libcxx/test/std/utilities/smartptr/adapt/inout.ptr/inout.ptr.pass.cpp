//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// [inout.ptr.t], class template inout_ptr_t
// template<class Smart, class Pointer, class... Args>
//   class inout_ptr_t;                                        // since c++23

// [inout.ptr], function template inout_ptr
// template<class Pointer = void, class Smart, class... Args>
//   auto inout_ptr(Smart& s, Args&&... args);                 // since c++23

#include <cassert>
#include <memory>

#include "../types.h"

// Test helpers.

void replace_int_p(int** pp) {
  assert(**pp == 90);
  delete *pp;
  *pp = new int{84};
}

void replace_int_p_with_nullptr(int** pp) {
  assert(**pp == 90);
  delete *pp;
  *pp = nullptr;
}

void replace_nullptr_with_int_p(int** pp) {
  assert(*pp == nullptr);
  *pp = new int{84};
}

void replace_int_void_p(void** pp) {
  assert(*(static_cast<int*>(*pp)) == 90);
  delete static_cast<int*>(*pp);
  *pp = new int{84};
}

void replace_int_void_p_with_nullptr(void** pp) {
  assert(*(static_cast<int*>(*pp)) == 90);
  delete static_cast<int*>(*pp);
  *pp = nullptr;
}

void replace_nullptr_with_int_void_p(void** pp) {
  assert(*pp == nullptr);
  *pp = new int{84};
}

void replace_SomeInt_p(SomeInt** pp) {
  auto si = **pp;
  assert(si.value == 90);
  delete static_cast<SomeInt*>(*pp);
  *pp = new SomeInt{9084};
}

void replace_SomeInt_void_p(void** pp) {
  assert(reinterpret_cast<SomeInt*>(*pp)->value == 90);
  delete static_cast<SomeInt*>(*pp);
  *pp = reinterpret_cast<void*>(new SomeInt{9084});
}

// Test `std::inout_ptr()` function.

void test_raw_ptr() {
  {
    auto rPtr = new int{90};

    replace_int_p(std::inout_ptr<int*>(rPtr));
    assert(*rPtr == 84);

    delete rPtr;
  }
  {
    auto rPtr = new int{90};

    replace_int_p_with_nullptr(std::inout_ptr<int*>(rPtr));
    assert(rPtr == nullptr);
  }
  {
    int* rPtr = nullptr;

    replace_nullptr_with_int_p(std::inout_ptr<int*>(rPtr));
    assert(*rPtr == 84);
    delete rPtr;
  }
  {
    auto rPtr = new int{90};

    replace_int_void_p(std::inout_ptr<int*>(rPtr));
    assert(*rPtr == 84);
    delete rPtr;
  }
  {
    auto rPtr = new int{90};

    replace_int_void_p_with_nullptr(std::inout_ptr<int*>(rPtr));
    assert(rPtr == nullptr);
  }
  {
    int* rPtr = nullptr;

    replace_nullptr_with_int_void_p(std::inout_ptr<int*>(rPtr));
    assert(*rPtr == 84);
    delete rPtr;
  }
  {
    auto* rPtr = new SomeInt{90};

    replace_SomeInt_p(std::inout_ptr(rPtr));
    assert(rPtr->value == 9084);
    delete rPtr;
  }
  {
    auto* rPtr = new SomeInt{90};

    replace_SomeInt_void_p(std::inout_ptr<SomeInt*>(rPtr));
    assert(rPtr->value == 9084);
    delete rPtr;
  }
}

void test_unique_ptr() {
  {
    auto uPtr = std::make_unique<int>(90);

    replace_int_p(std::inout_ptr(uPtr));
    assert(*uPtr == 84);
  }
  {
    std::unique_ptr<int> uPtr;

    replace_nullptr_with_int_p(std::inout_ptr(uPtr));
    assert(*uPtr == 84);
  }
  {
    auto uPtr = std::make_unique<int>(90);

    replace_int_void_p(std::inout_ptr(uPtr));
    assert(*uPtr == 84);
  }
  {
    std::unique_ptr<int> uPtr;

    replace_nullptr_with_int_void_p(std::inout_ptr(uPtr));
    assert(*uPtr == 84);
  }
  {
    auto uPtr = std::make_unique<SomeInt>(90);

    replace_SomeInt_p(std::inout_ptr(uPtr));
    assert(uPtr->value == 9084);
  }
  {
    auto uPtr = std::make_unique<SomeInt>(90);

    replace_SomeInt_void_p(std::inout_ptr<SomeInt*>(uPtr));
    assert(uPtr->value == 9084);
  }
}

void test_custom_ptr() {
  // ConstructiblePtr
  {
    ConstructiblePtr<int> cPtr(new int{90});

    replace_int_p(std::inout_ptr(cPtr));
    assert(cPtr == 84);
  }
  // ResettablePtr
  {
    ResettablePtr<int> rPtr(new int{90});

    replace_int_p(std::inout_ptr(rPtr));
    assert(rPtr == 84);
  }
  // NonConstructiblePtr
  {
    NonConstructiblePtr<int> nPtr;
    nPtr.reset(new int{90});

    replace_int_p(std::inout_ptr(nPtr));
    assert(nPtr == 84);
  }
}

int main(int, char**) {
  test_raw_ptr();
  test_unique_ptr();
  test_custom_ptr();

  return 0;
}
