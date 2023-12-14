//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// [out.ptr], function template out_ptr
// template<class Pointer = void, class Smart, class... Args>
//   auto out_ptr(Smart& s, Args&&... args);                   // since c++23

#include <cassert>
#include <memory>

#include "../types.h"

// Test helpers.

void get_int_p(int** pp) { *pp = new int{84}; }

void get_int_p_nullptr(int** pp) { *pp = nullptr; }

void get_int_void_p(void** pp) { *(reinterpret_cast<int**>(pp)) = new int{84}; }

void get_int_void_p_nullptr(void** pp) { *pp = nullptr; }

void get_SomeInt_p(SomeInt** pp) { *pp = new SomeInt{84}; }

void get_SomeInt_void_p(void** pp) { *pp = reinterpret_cast<int*>(new int{84}); }

// Test `std::out_ptr()` function.

void test_raw_ptr() {
  {
    auto n{90};
    auto rPtr = &n;

    get_int_p(std::inout_ptr<int*>(rPtr));
    assert(*rPtr == 84);
    delete rPtr;

    get_int_p_nullptr(std::out_ptr<int*>(rPtr));
    assert(rPtr == nullptr);

    get_int_void_p(std::inout_ptr<int*>(rPtr));
    assert(*rPtr == 84);
    delete rPtr;

    get_int_void_p_nullptr(std::out_ptr<int*>(rPtr));
    assert(rPtr == nullptr);
  }
  {
    SomeInt si{90};
    auto* rPtr = &si;

    get_SomeInt_p(std::out_ptr(rPtr));
    assert(rPtr->value == 84);
    delete rPtr;
  }
  {
    SomeInt si{90};
    auto* rPtr = &si;

    get_SomeInt_void_p(std::out_ptr<SomeInt*>(rPtr));
    assert(rPtr->value == 84);
    delete rPtr;
  }
}

void test_shared_ptr() {
  {
    auto sPtr = std::make_shared<int>(90);

    get_int_p(std::out_ptr(sPtr, [](auto* p) { delete p; }));
    assert(*sPtr == 84);

    sPtr.reset(new int(90));

    get_int_void_p(std::out_ptr(sPtr, [](auto* p) { delete p; }));
    assert(*sPtr == 84);
  }
  {
    auto sPtr = std::make_shared<SomeInt>(90);

    get_SomeInt_p(std::out_ptr(sPtr, [](auto* p) { delete p; }));
    assert(sPtr->value == 84);
  }
  {
    auto sPtr = std::make_shared<SomeInt>(90);

    get_SomeInt_void_p(std::out_ptr<SomeInt*>(sPtr, [](auto* p) { delete p; }));
    assert(sPtr->value == 84);
  }
}

void test_unique_ptr() {
  {
    auto uPtr = std::make_unique<int>(90);

    get_int_p(std::out_ptr(uPtr));
    assert(*uPtr == 84);

    uPtr.reset(new int{90});

    get_int_void_p(std::out_ptr(uPtr));
    assert(*uPtr == 84);
  }
  {
    auto uPtr = std::make_unique<SomeInt>(90);

    get_SomeInt_p(std::out_ptr(uPtr));
    assert(uPtr->value == 84);
  }
  {
    auto uPtr = std::make_unique<SomeInt>(90);

    get_SomeInt_void_p(std::out_ptr<SomeInt*>(uPtr));
    assert(uPtr->value == 84);
  }
}

void test_custom_ptr() {
  // ConstructiblePtr
  {
    ConstructiblePtr<int> cPtr(new int{90});

    get_int_p(std::out_ptr(cPtr));
    assert(cPtr == 84);
  }
  // ResettablePtr
  {
    ResettablePtr<int> rPtr(new int{90});

    get_int_p(std::out_ptr(rPtr));
    assert(rPtr == 84);
  }
  // NonConstructiblePtr
  {
    NonConstructiblePtr<int> nPtr;
    nPtr.reset(new int{90});

    get_int_p(std::out_ptr(nPtr));
    assert(nPtr == 84);
  }
}

int main(int, char**) {
  test_raw_ptr();
  test_shared_ptr();
  test_unique_ptr();
  test_custom_ptr();

  return 0;
}
