//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// UNSUPPORTED: c++03, no-exceptions

// TODO:
// - throwing upon moving;
// - initializer lists;
// - throwing when constructing the element in place.

// forward_list(size_type n, const value_type& v);
// forward_list(size_type n, const value_type& v, const allocator_type& a);
// template <class InputIterator>
//     forward_list(InputIterator first, InputIterator last);
// template <class InputIterator>
//     forward_list(InputIterator first, InputIterator last, const allocator_type& a);
// forward_list(const forward_list& x);
// forward_list(const forward_list& x, const allocator_type& a);
// template<container-compatible-range<T> R>
//     forward_list(from_range_t, R&& rg, const Allocator& = Allocator()); // C++23
//
// forward_list& operator=(const forward_list& x);
//
// template <class InputIterator>
//     void assign(InputIterator first, InputIterator last);
// void assign(size_type n, const value_type& v);
// template<container-compatible-range<T> R>
//     void assign_range(R&& rg); // C++23
//
// void push_front(const value_type& v);
//  template<container-compatible-range<T> R>
//    void prepend_range(R&& rg); // C++23
//
// iterator insert_after(const_iterator p, const value_type& v);
// iterator insert_after(const_iterator p, size_type n, const value_type& v);
// template <class InputIterator>
//     iterator insert_after(const_iterator p,
//                           InputIterator first, InputIterator last);
//  template<container-compatible-range<T> R>
//     iterator insert_range_after(const_iterator position, R&& rg); // C++23
//
// void resize(size_type n, const value_type& v);

#include <forward_list>

#include <cassert>
#include "../../exception_safety_helpers.h"
#include "test_macros.h"

#if TEST_STD_VER >= 23
#include <ranges>
#endif

int main(int, char**) {
  {
    constexpr int ThrowOn = 1;
    constexpr int Size = 1;
    using T = ThrowingCopy<ThrowOn>;

    // void push_front(const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*){
      std::forward_list<T> c;
      c.push_front(*from);
    });

    // iterator insert_after(const_iterator p, const value_type& v);
    test_exception_safety_throwing_copy</*ThrowOn=*/1, Size>([](T* from, T*){
      std::forward_list<T> c;
      c.insert_after(c.before_begin(), *from);
    });
  }

  {
    constexpr int ThrowOn = 3;
    constexpr int Size = 5;
    using T = ThrowingCopy<ThrowOn>;
    using C = std::forward_list<T>;
    using Alloc = std::allocator<T>;

    // forward_list(size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*){
      std::forward_list<T> c(Size, *from);
      (void)c;
    });

    // forward_list(size_type n, const value_type& v, const allocator_type& a);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*){
      std::forward_list<T> c(Size, *from, Alloc());
      (void)c;
    });

    // template <class InputIterator>
    //     forward_list(InputIterator first, InputIterator last);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to){
      std::forward_list<T> c(from, to);
      (void)c;
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //     forward_list(from_range_t, R&& rg, const Allocator& = Allocator()); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to){
      {
        std::forward_list<T> c(std::from_range, std::ranges::subrange(from, to));
        (void)c;
      }

      {
        std::forward_list<T> c(std::from_range, std::ranges::subrange(from, to), Alloc());
        (void)c;
      }
    });
#endif

    // template <class InputIterator>
    //     forward_list(InputIterator first, InputIterator last, const allocator_type& a);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to){
      std::forward_list<T> c(from, to, Alloc());
      (void)c;
    });

    // forward_list(const forward_list& x);
    test_exception_safety_throwing_copy_container<C, ThrowOn, Size>([](C&& in) {
      std::forward_list<T> c(in);
      (void)c;
    });

    // forward_list(const forward_list& x, const allocator_type& a);
    test_exception_safety_throwing_copy_container<C, ThrowOn, Size>([](C&& in) {
      std::forward_list<T> c(in, Alloc());
      (void)c;
    });

    // forward_list& operator=(const forward_list& x);
    test_exception_safety_throwing_copy_container<C, ThrowOn, Size>([](C&& in) {
      std::forward_list<T> c;
      c = in;
    });

    // template <class InputIterator>
    //     void assign(InputIterator first, InputIterator last);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::forward_list<T> c;
      c.assign(from, to);
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //     void assign_range(R&& rg); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::forward_list<T> c;
      c.assign_range(std::ranges::subrange(from, to));
    });
#endif

    // void assign(size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::forward_list<T> c;
      c.assign(Size, *from);
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //   void prepend_range(R&& rg); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::forward_list<T> c;
      c.prepend_range(std::ranges::subrange(from, to));
    });
#endif

    // iterator insert_after(const_iterator p, size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::forward_list<T> c;
      c.insert_after(c.before_begin(), Size, *from);
    });

    // template <class InputIterator>
    //     iterator insert_after(const_iterator p,
    //                           InputIterator first, InputIterator last);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::forward_list<T> c;
      c.insert_after(c.before_begin(), from, to);
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //     iterator insert_range_after(const_iterator position, R&& rg); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::forward_list<T> c;
      c.insert_range_after(c.before_begin(), std::ranges::subrange(from, to));
    });
#endif

    // void resize(size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::forward_list<T> c;
      c.resize(Size, *from);
    });
  }

  return 0;
}
