//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_SHARED_PTR_UNIQUE

// <memory>

// Check that functions are marked [[nodiscard]]

#include <memory>
#include <utility>

#include "test_macros.h"

void test() {
  { // [unique.ptr]
    std::unique_ptr<int> uPtr;

    *uPtr;              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    uPtr.get();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    uPtr.get_deleter(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    const std::unique_ptr<int> cuPtr;
    cuPtr.get_deleter(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 14
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_unique<int>(94);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_unique<int[]>(82);
#endif
#if TEST_STD_VER >= 20
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_unique_for_overwrite<int>();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_unique_for_overwrite<int[]>(5);
#endif
  }
  { // [util.sharedptr]
    std::shared_ptr<int[]> sPtr;

    sPtr.get();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    *sPtr;            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sPtr.use_count(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER <= 20
    sPtr.unique(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sPtr.owner_before(std::shared_ptr<int>());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sPtr.owner_before(std::weak_ptr<int>());
#if TEST_STD_VER >= 17
    sPtr[0]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::allocate_shared<int>(std::allocator<int>(), 5);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_shared<int>();
#if TEST_STD_VER >= 20
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::allocate_shared_for_overwrite<int>(std::allocator<int>());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_shared_for_overwrite<int>();

    // Bounded array variants
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::allocate_shared<int[5]>(std::allocator<int>());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::allocate_shared<int[5]>(std::allocator<int>(), 5);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::allocate_shared_for_overwrite<int[5]>(std::allocator<int>());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_shared<int[5]>();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_shared<int[5]>(94);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_shared_for_overwrite<int[5]>();

    // Unbounded array variants
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::allocate_shared<int[]>(std::allocator<int>(), 5);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::allocate_shared<int[]>(std::allocator<int>(), 5, 94);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::allocate_shared_for_overwrite<int[]>(std::allocator<int>(), 5);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_shared<int[]>(5);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_shared<int[]>(5, 82);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_shared_for_overwrite<int[]>(5);
#endif

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::static_pointer_cast<int[]>(sPtr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::static_pointer_cast<int[]>(std::move(sPtr));
    class Empty {};
    std::shared_ptr<Empty> dsPtr;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::dynamic_pointer_cast<Empty>(dsPtr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::dynamic_pointer_cast<Empty>(std::move(dsPtr));
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::const_pointer_cast<int[]>(sPtr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::const_pointer_cast<int[]>(std::move(sPtr));
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reinterpret_pointer_cast<int[]>(sPtr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reinterpret_pointer_cast<int[]>(std::move(sPtr));
#if !defined(TEST_HAS_NO_RTTI)
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get_deleter<int[]>(sPtr);
#endif
  }
  { // [util.smartptr.weak]
    std::weak_ptr<int> wPtr;

    wPtr.use_count(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    wPtr.expired();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    wPtr.lock();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wPtr.owner_before(std::weak_ptr<int>());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    wPtr.owner_before(std::shared_ptr<int>());
  }
  { // [util.smartptr.enab]
    class EnableShared : public std::enable_shared_from_this<EnableShared> {};
    EnableShared es;
    const EnableShared ces;

    es.shared_from_this(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ces.shared_from_this(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 17
    es.weak_from_this();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ces.weak_from_this(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
  }
#if TEST_STD_VER >= 23
  { // [smartptr.adapt]
    std::unique_ptr<int> uPtr;
    // [inout.ptr]
    std::inout_ptr(uPtr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // [out.ptr]
    std::out_ptr(uPtr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif
}
