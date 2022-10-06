//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template<class T, class Alloc, class... Args>
//   constexpr T uninitialized_construct_using_allocator(const Alloc& alloc, Args&&... args);

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <concepts>
#include <memory>
#include <tuple>
#include <utility>

#include "common.h"
#include "test_allocator.h"

constexpr bool test() {
  Alloc a(12);
  {
    auto* ptr                             = std::allocator<UsesAllocArgT>{}.allocate(1);
    std::same_as<UsesAllocArgT*> auto ret = std::uninitialized_construct_using_allocator(ptr, a);
    assert(ret == ptr);
    assert(ret->allocator_constructed_);
    assert(&ret->alloc_ == &a);
    std::allocator<UsesAllocArgT>{}.deallocate(ptr, 1);
  }
  {
    auto* ptr                             = std::allocator<UsesAllocLast>{}.allocate(1);
    std::same_as<UsesAllocLast*> auto ret = std::uninitialized_construct_using_allocator(ptr, a);
    assert(ret->allocator_constructed_);
    assert(&ret->alloc_ == &a);
    std::allocator<UsesAllocLast>{}.deallocate(ptr, 1);
  }
  {
    auto* ptr                                 = std::allocator<NotAllocatorAware>{}.allocate(1);
    std::same_as<NotAllocatorAware*> auto ret = std::uninitialized_construct_using_allocator(ptr, a);
    assert(!ret->allocator_constructed_);
    std::allocator<NotAllocatorAware>{}.deallocate(ptr, 1);
  }
  {
    auto* ptr = std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.allocate(1);
    std::same_as<std::pair<UsesAllocArgT, UsesAllocLast>*> auto ret =
        std::uninitialized_construct_using_allocator(ptr, a, std::piecewise_construct, std::tuple<>{}, std::tuple<>{});
    assert(ret->first.allocator_constructed_);
    assert(&ret->first.alloc_ == &a);
    assert(ret->second.allocator_constructed_);
    assert(&ret->second.alloc_ == &a);
    std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.deallocate(ptr, 1);
  }
  {
    auto* ptr = std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.allocate(1);
    std::same_as<std::pair<UsesAllocArgT, UsesAllocLast>*> auto ret =
        std::uninitialized_construct_using_allocator(ptr, a);
    assert(ret->first.allocator_constructed_);
    assert(&ret->first.alloc_ == &a);
    assert(ret->second.allocator_constructed_);
    assert(&ret->second.alloc_ == &a);
    std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.deallocate(ptr, 1);
  }
  {
    int val   = 0;
    auto* ptr = std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.allocate(1);
    std::same_as<std::pair<UsesAllocArgT, UsesAllocLast>*> auto ret =
        std::uninitialized_construct_using_allocator(ptr, a, val, val);
    assert(ret->first.allocator_constructed_);
    assert(&ret->first.alloc_ == &a);
    assert(ret->first.ref_type_ == RefType::LValue);
    assert(ret->first.val_ptr_ == &val);
    assert(ret->second.allocator_constructed_);
    assert(&ret->second.alloc_ == &a);
    assert(ret->second.ref_type_ == RefType::LValue);
    assert(ret->second.val_ptr_ == &val);
    std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.deallocate(ptr, 1);
  }
  {
    int val   = 0;
    auto* ptr = std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.allocate(1);
    std::same_as<std::pair<UsesAllocArgT, UsesAllocLast>*> auto ret =
        std::uninitialized_construct_using_allocator(ptr, a, std::move(val), std::move(val));
    assert(ret->first.allocator_constructed_);
    assert(&ret->first.alloc_ == &a);
    assert(ret->first.ref_type_ == RefType::RValue);
    assert(ret->first.val_ptr_ == &val);
    assert(ret->second.allocator_constructed_);
    assert(&ret->second.alloc_ == &a);
    assert(ret->second.ref_type_ == RefType::RValue);
    assert(ret->second.val_ptr_ == &val);
    std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.deallocate(ptr, 1);
  }
#if TEST_STD_VER >= 23
  {
    std::pair p{0, 0};

    auto* ptr = std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.allocate(1);
    std::same_as<std::pair<UsesAllocArgT, UsesAllocLast>*> auto ret =
        std::uninitialized_construct_using_allocator(ptr, a, p);
    assert(ret->first.allocator_constructed_);
    assert(&ret->first.alloc_ == &a);
    assert(ret->first.ref_type_ == RefType::LValue);
    assert(ret->first.val_ptr_ == &p.first);
    assert(ret->second.allocator_constructed_);
    assert(&ret->second.alloc_ == &a);
    assert(ret->second.ref_type_ == RefType::LValue);
    assert(ret->second.val_ptr_ == &p.second);
    std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.deallocate(ptr, 1);
  }
#endif
  {
    std::pair p{0, 0};
    auto* ptr = std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.allocate(1);
    std::same_as<std::pair<UsesAllocArgT, UsesAllocLast>*> auto ret =
        std::uninitialized_construct_using_allocator(ptr, a, std::as_const(p));
    assert(ret->first.allocator_constructed_);
    assert(&ret->first.alloc_ == &a);
    assert(ret->first.ref_type_ == RefType::ConstLValue);
    assert(ret->first.val_ptr_ == &p.first);
    assert(ret->second.allocator_constructed_);
    assert(&ret->second.alloc_ == &a);
    assert(ret->second.ref_type_ == RefType::ConstLValue);
    assert(ret->second.val_ptr_ == &p.second);
    std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.deallocate(ptr, 1);
  }
  {
    std::pair p{0, 0};
    auto* ptr = std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.allocate(1);
    std::same_as<std::pair<UsesAllocArgT, UsesAllocLast>*> auto ret =
        std::uninitialized_construct_using_allocator(ptr, a, std::move(p));
    assert(ret->first.allocator_constructed_);
    assert(&ret->first.alloc_ == &a);
    assert(ret->first.ref_type_ == RefType::RValue);
    assert(ret->first.val_ptr_ == &p.first);
    assert(ret->second.allocator_constructed_);
    assert(&ret->second.alloc_ == &a);
    assert(ret->second.ref_type_ == RefType::RValue);
    assert(ret->second.val_ptr_ == &p.second);
    std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.deallocate(ptr, 1);
  }
#if TEST_STD_VER >= 23
  {
    std::pair p{0, 0};
    auto* ptr = std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.allocate(1);
    std::same_as<std::pair<UsesAllocArgT, UsesAllocLast>*> auto ret =
        std::uninitialized_construct_using_allocator(ptr, a, std::move(std::as_const(p)));
    assert(ret->first.allocator_constructed_);
    assert(&ret->first.alloc_ == &a);
    assert(ret->first.ref_type_ == RefType::ConstRValue);
    assert(ret->first.val_ptr_ == &p.first);
    assert(ret->second.allocator_constructed_);
    assert(&ret->second.alloc_ == &a);
    assert(ret->second.ref_type_ == RefType::ConstRValue);
    assert(ret->second.val_ptr_ == &p.second);
    std::allocator<std::pair<UsesAllocArgT, UsesAllocLast>>{}.deallocate(ptr, 1);
  }
#endif
  {
    ConvertibleToPair ctp;
    auto* ptr                                   = std::allocator<std::pair<int, int>>{}.allocate(1);
    std::same_as<std::pair<int, int>*> auto ret = std::uninitialized_construct_using_allocator(ptr, a, ctp);
    assert(ret == ptr);
    assert(ret->first == 1);
    assert(ret->second == 2);
    std::allocator<std::pair<int, int>>{}.deallocate(ptr, 1);
  }
  {
    ConvertibleToPair ctp;
    auto* ptr                                   = std::allocator<std::pair<int, int>>{}.allocate(1);
    std::same_as<std::pair<int, int>*> auto ret = std::uninitialized_construct_using_allocator(ptr, a, std::move(ctp));
    assert(ret == ptr);
    assert(ret->first == 1);
    assert(ret->second == 2);
    std::allocator<std::pair<int, int>>{}.deallocate(ptr, 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
}
