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
#include <utility>

#include "../types.h"

// Test updating an `out_ptr_t`-managed pointer for an API with a non-void pointer type.
// The API returns a new valid object.
void test_get_int_p() {
  auto get_int_p = [](int** pp) { *pp = new int{84}; };

  // raw pointer
  {
    int* rPtr;

    get_int_p(std::out_ptr<int*>(rPtr));
    assert(*rPtr == 84);

    delete rPtr;
  }

  // std::unique_ptr
  {
    std::unique_ptr<int> uPtr;

    get_int_p(std::out_ptr(uPtr));
    assert(*uPtr == 84);
  }

  {
    MoveOnlyDeleter<int> del;
    std::unique_ptr<int, MoveOnlyDeleter<int>> uPtr;

    get_int_p(std::out_ptr(uPtr, std::move(del)));
    assert(*uPtr == 84);
    assert(uPtr.get_deleter().wasMoveInitilized == true);
  }

  // std::shared_ptr
  {
    std::shared_ptr<int> sPtr;

    get_int_p(std::out_ptr(sPtr, [](auto* p) {
      assert(*p == 84);

      delete p;
    }));
    assert(*sPtr == 84);
  }

  // pointer-like ConstructiblePtr
  {
    ConstructiblePtr<int> cPtr;

    get_int_p(std::out_ptr(cPtr));
    assert(cPtr == 84);
  }

  // pointer-like ResettablePtr
  {
    ResettablePtr<int> rPtr{nullptr};

    get_int_p(std::out_ptr(rPtr));
    assert(rPtr == 84);
  }

  // NonConstructiblePtr
  {
    NonConstructiblePtr<int> nPtr;

    get_int_p(std::out_ptr(nPtr));
    assert(nPtr == 84);
  }
}

// Test updating an `out_ptr_t`-managed pointer for an API with a non-void pointer type.
// The API returns `nullptr`.
void test_get_int_p_nullptr() {
  auto get_int_p_nullptr = [](int** pp) { *pp = nullptr; };
  // raw pointer
  {
    int* rPtr;

    get_int_p_nullptr(std::out_ptr<int*>(rPtr));
    assert(rPtr == nullptr);

    delete rPtr;
  }

  // std::unique_ptr
  {
    std::unique_ptr<int> uPtr;

    get_int_p_nullptr(std::out_ptr(uPtr));
    assert(uPtr == nullptr);
  }

  // std::shared_ptr
  {
    std::shared_ptr<int> sPtr;

    get_int_p_nullptr(std::out_ptr(sPtr, [](auto* p) {
      assert(p == nullptr);

      delete p;
    }));
    assert(sPtr == nullptr);
  }
}

// Test updating an `out_ptr_t`-managed pointer for an API with a void pointer type.
// The API returns a new valid object.
void test_get_int_void_p() {
  auto get_int_void_p = [](void** pp) { *(reinterpret_cast<int**>(pp)) = new int{84}; };

  // raw pointer
  {
    int* rPtr;

    get_int_void_p(std::out_ptr(rPtr));
    assert(*rPtr == 84);

    delete rPtr;
  }

  // std::unique_ptr
  {
    std::unique_ptr<int> uPtr;

    get_int_void_p(std::out_ptr(uPtr));
    assert(*uPtr == 84);
  }

  // std::shared_ptr
  {
    std::shared_ptr<int> sPtr;

    get_int_void_p(std::out_ptr(sPtr, [](auto* p) {
      assert(*p == 84);

      delete p;
    }));
    assert(*sPtr == 84);
  }

  // pointer-like ConstructiblePtr
  {
    ConstructiblePtr<int> cPtr;

    get_int_void_p(std::out_ptr(cPtr));
    assert(cPtr == 84);
  }

  // pointer-like ResettablePtr
  {
    ResettablePtr<int> rPtr{nullptr};

    get_int_void_p(std::out_ptr(rPtr));
    assert(rPtr == 84);
  }

  // NonConstructiblePtr
  {
    NonConstructiblePtr<int> nPtr;

    get_int_void_p(std::out_ptr(nPtr));
    assert(nPtr == 84);
  }
}

// Test updating an `out_ptr_t`-managed pointer for an API with a void pointer type.
// The API returns `nullptr`.
void test_get_int_void_p_nullptr() {
  auto get_int_void_p_nullptr = [](void** pp) { *pp = nullptr; };

  // raw pointer
  {
    int* rPtr;

    get_int_void_p_nullptr(std::out_ptr<int*>(rPtr));
    assert(rPtr == nullptr);

    delete rPtr;
  }

  // std::unique_ptr
  {
    std::unique_ptr<int> uPtr;

    get_int_void_p_nullptr(std::out_ptr(uPtr));
    assert(uPtr == nullptr);
  }

  // std::shared_ptr
  {
    std::shared_ptr<int> sPtr;

    get_int_void_p_nullptr(std::out_ptr(sPtr, [](auto* p) {
      assert(p == nullptr);

      delete p;
    }));
    assert(sPtr == nullptr);
  }
}

int main(int, char**) {
  test_get_int_p();
  test_get_int_p_nullptr();
  test_get_int_void_p();
  test_get_int_void_p_nullptr();

  return 0;
}
