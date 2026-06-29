//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// UNSUPPORTED: c++03, no-exceptions

// TODO:
// - deque(deque&&, const allocator_type&)

// deque(size_type n);
// deque(size_type n, const allocator_type& a);
// deque(size_type n, const value_type& v);
// deque(size_type n, const value_type& v, const allocator_type& a);
// template <class InputIterator>
//     deque(InputIterator first, InputIterator last);
// template <class InputIterator>
//     deque(InputIterator first, InputIterator last, const allocator_type& a);
// template<container-compatible-range<T> R>
//     deque(from_range_t, R&& rg, const Allocator& = Allocator()); // C++23
// deque(const deque& x);
// deque(const deque& x, const allocator_type& a);
//
// deque& operator=(const deque& x);
//
// template <class InputIterator>
//     void assign(InputIterator first, InputIterator last);
// void assign(size_type n, const value_type& v);
// template<container-compatible-range<T> R>
//     void assign_range(R&& rg); // C++23
//
// template<container-compatible-range<T> R>
//   void prepend_range(R&& rg); // C++23
// void push_back(const value_type& x);
// template<container-compatible-range<T> R>
//   void append_range(R&& rg); // C++23
// void push_front(const value_type& v);
//
// iterator insert(const_iterator p, const value_type& v);
// iterator insert(const_iterator p, size_type n, const value_type& v);
// template <class InputIterator>
//     iterator insert(const_iterator p,
//                     InputIterator first, InputIterator last);
// template<container-compatible-range<T> R>
//     iterator insert_range(const_iterator position, R&& rg); // C++23
//
// void resize(size_type n, const value_type& v);

#include <deque>

#include <cassert>
#include "../../exception_safety_helpers.h"
#include "test_macros.h"

#if TEST_STD_VER >= 23
#  include <ranges>
#endif

int main(int, char**) {
  {
    constexpr int ThrowOn = 1;
    constexpr int Size    = 1;
    using T               = ThrowingCopy<ThrowOn>;

    // void push_front(const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::deque<T> c;
      c.push_front(*from);
    });

    // void push_back(const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::deque<T> c;
      c.push_back(*from);
    });

    // iterator insert(const_iterator p, const value_type& v);
    test_exception_safety_throwing_copy</*ThrowOn=*/1, Size>([](T* from, T*) {
      std::deque<T> c;
      c.insert(c.end(), *from);
    });
  }

  {
    constexpr int ThrowOn = 3;
    constexpr int Size    = 5;
    using T               = ThrowingDefault<ThrowOn>;
    using Alloc           = std::allocator<T>;

    // deque(size_type n);
    test_exception_safety_throwing_default<ThrowOn, Size>([](size_t n) {
      std::deque<T> c(n);
      (void)c;
    });

    // deque(size_type n, const allocator_type& a);
    test_exception_safety_throwing_default<ThrowOn, Size>([](size_t n) {
      std::deque<T> c(n, Alloc());
      (void)c;
    });
  }

  {
    constexpr int ThrowOn = 3;
    constexpr int Size    = 5;
    using T               = ThrowingCopy<ThrowOn>;
    using C               = std::deque<T>;
    using Alloc           = std::allocator<T>;

    std::initializer_list<T> il = {1, 2, 3, 4, 5};

    // deque(size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::deque<T> c(Size, *from);
      (void)c;
    });

    // deque(size_type n, const value_type& v, const allocator_type& a);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::deque<T> c(Size, *from, Alloc());
      (void)c;
    });

    // template <class InputIterator>
    //     deque(InputIterator first, InputIterator last);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::deque<T> c(from, to);
      (void)c;
    });

    // template <class InputIterator>
    //     deque(InputIterator first, InputIterator last, const allocator_type& a);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::deque<T> c(from, to, Alloc());
      (void)c;
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //     deque(from_range_t, R&& rg, const Allocator& = Allocator()); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      {
        std::deque<T> c(std::from_range, std::ranges::subrange(from, to));
        (void)c;
      }

      {
        std::deque<T> c(std::from_range, std::ranges::subrange(from, to), Alloc());
        (void)c;
      }
    });
#endif

    // template <class InputIterator>
    //     deque(InputIterator first, InputIterator last, const allocator_type& a);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::deque<T> c(from, to, Alloc());
      (void)c;
    });

    // deque(const deque& x);
    test_exception_safety_throwing_copy_container<C, ThrowOn, Size>([](C&& in) {
      std::deque<T> c(in);
      (void)c;
    });

    // deque(const deque& x, const allocator_type& a);
    test_exception_safety_throwing_copy_container<C, ThrowOn, Size>([](C&& in) {
      std::deque<T> c(in, Alloc());
      (void)c;
    });

    // deque(initializer_list<value_type> il);
    test_exception_safety_throwing_copy<ThrowOn, Size>([&il](T*, T*) {
      std::deque<T> c(il);
      (void)c;
    });

    // deque(initializer_list<value_type> il, const allocator_type& a);
    test_exception_safety_throwing_copy<ThrowOn, Size>([&il](T*, T*) {
      std::deque<T> c(il, Alloc());
      (void)c;
    });

    // deque& operator=(const deque& x);
    test_exception_safety_throwing_copy_container<C, ThrowOn, Size>([](C&& in) {
      std::deque<T> c;
      c = in;
    });

    // deque& operator=(initializer_list<value_type> il);
    test_exception_safety_throwing_copy<ThrowOn, Size>([&il](T*, T*) {
      std::deque<T> c;
      c = il;
    });

    // template <class InputIterator>
    //     void assign(InputIterator first, InputIterator last);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::deque<T> c;
      c.assign(from, to);
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //     void assign_range(R&& rg); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::deque<T> c;
      c.assign_range(std::ranges::subrange(from, to));
    });
#endif

    // void assign(size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::deque<T> c;
      c.assign(Size, *from);
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //   void prepend_range(R&& rg); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::deque<T> c;
      c.prepend_range(std::ranges::subrange(from, to));
    });

    // template<container-compatible-range<T> R>
    //   void append_range(R&& rg); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::deque<T> c;
      c.append_range(std::ranges::subrange(from, to));
    });
#endif

    // iterator insert(const_iterator p, size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::deque<T> c;
      c.insert(c.end(), Size, *from);
    });

    // template <class InputIterator>
    //     iterator insert(const_iterator p,
    //                     InputIterator first, InputIterator last);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::deque<T> c;
      c.insert(c.end(), from, to);
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //     iterator insert_range(const_iterator position, R&& rg); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::deque<T> c;
      c.insert_range(c.end(), std::ranges::subrange(from, to));
    });
#endif

    // void resize(size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::deque<T> c;
      c.resize(Size, *from);
    });
  }

  return 0;
}
