//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// [inout.ptr], function template inout_ptr
// template<class Pointer = void, class Smart, class... Args>
//   auto inout_ptr(Smart& s, Args&&... args);                 // since c++23

#include <cassert>
#include <memory>
#include <utility>

#include "../types.h"

// Test updating the ownership of an `inout_ptr_t`-managed pointer for an API with a non-void pointer type.
// The API returns a new valid object.
void test_replace_int_p() {
  auto replace_int_p = [](int** pp) {
    assert(**pp == 90);

    delete *pp;
    *pp = new int{84};
  };

  // raw pointer
  {
    auto rPtr = new int{90};

    replace_int_p(std::inout_ptr<int*>(rPtr));
    assert(*rPtr == 84);

    delete rPtr;
  }

  // std::unique_ptr
  {
    auto uPtr = std::make_unique<int>(90);

    replace_int_p(std::inout_ptr(uPtr));
    assert(*uPtr == 84);
  }

  {
    MoveOnlyDeleter<int> del;
    std::unique_ptr<int, MoveOnlyDeleter<int>> uPtr{new int{90}};

    replace_int_p(std::inout_ptr(uPtr, std::move(del)));
    assert(*uPtr == 84);
    assert(uPtr.get_deleter().wasMoveInitilized == true);
  }

  // pointer-like ConstructiblePtr
  {
    ConstructiblePtr<int> cPtr(new int{90});

    replace_int_p(std::inout_ptr(cPtr));
    assert(cPtr == 84);
  }

  // pointer-like ResettablePtr
  {
    ResettablePtr<int> rPtr(new int{90});

    replace_int_p(std::inout_ptr(rPtr));
    assert(rPtr == 84);
  }

  // pointer-like NonConstructiblePtr
  {
    NonConstructiblePtr<int> nPtr;
    nPtr.reset(new int{90});

    replace_int_p(std::inout_ptr(nPtr));
    assert(nPtr == 84);
  }
}

// Test updating the ownership of an `inout_ptr_t`-managed pointer for an API with a non-void pointer type.
// The API returns `nullptr`.
void test_replace_int_p_with_nullptr() {
  auto replace_int_p_with_nullptr = [](int** pp) -> void {
    assert(**pp == 90);

    delete *pp;
    *pp = nullptr;
  };

  // raw pointer
  {
    // LWG-3897 inout_ptr will not update raw pointer to null
    auto rPtr = new int{90};

    replace_int_p_with_nullptr(std::inout_ptr<int*>(rPtr));
    assert(rPtr == nullptr);
  }

  // std::unique_ptr
  {
    auto uPtr = std::make_unique<int>(90);

    replace_int_p_with_nullptr(std::inout_ptr(uPtr));
    assert(uPtr == nullptr);
  }
}

// Test updating the ownership of an `inout_ptr_t`-managed pointer for an API with a void pointer type.
// The API returns a new valid object.
void test_replace_int_void_p() {
  auto replace_int_void_p = [](void** pp) {
    assert(*(static_cast<int*>(*pp)) == 90);

    delete static_cast<int*>(*pp);
    *pp = new int{84};
  };

  // raw pointer
  {
    auto rPtr = new int{90};

    replace_int_void_p(std::inout_ptr<int*>(rPtr));
    assert(*rPtr == 84);

    delete rPtr;
  }

  // std::unique_ptr
  {
    auto uPtr = std::make_unique<int>(90);

    replace_int_void_p(std::inout_ptr(uPtr));
    assert(*uPtr == 84);
  }
}

// Test updating the ownership of an `inout_ptr_t`-managed pointer for an API with a non-void pointer type.
// The API returns `nullptr`.
void test_replace_int_void_p_with_nullptr() {
  auto replace_int_void_p_with_nullptr = [](void** pp) {
    assert(*(static_cast<int*>(*pp)) == 90);

    delete static_cast<int*>(*pp);
    *pp = nullptr;
  };

  // raw pointer
  {
    auto rPtr = new int{90};

    replace_int_void_p_with_nullptr(std::inout_ptr<int*>(rPtr));
    assert(rPtr == nullptr);
  }

  // std::unique_ptr
  {
    auto uPtr = std::make_unique<int>(90);

    replace_int_void_p_with_nullptr(std::inout_ptr(uPtr));
    assert(uPtr == nullptr);
  }
}

// Test updating the ownership of an `inout_ptr_t`-managed pointer for an API with a void pointer type.
// The API returns a new valid object.
void test_replace_nullptr_with_int_p() {
  auto replace_nullptr_with_int_p = [](int** pp) {
    assert(*pp == nullptr);

    *pp = new int{84};
  };

  // raw pointer
  {
    int* rPtr = nullptr;

    replace_nullptr_with_int_p(std::inout_ptr<int*>(rPtr));
    assert(*rPtr == 84);

    delete rPtr;
  }

  // std::unique_ptr
  {
    std::unique_ptr<int> uPtr;

    replace_nullptr_with_int_p(std::inout_ptr(uPtr));
    assert(*uPtr == 84);
  }
}

int main(int, char**) {
  test_replace_int_p();
  test_replace_int_p_with_nullptr();
  test_replace_int_void_p();
  test_replace_int_void_p_with_nullptr();
  test_replace_nullptr_with_int_p();

  return 0;
}
