//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// UNSUPPORTED: c++03, no-exceptions

// TODO:
// - throwing upon moving;
// - initializer lists;
// - throwing when constructing the element in place.

// list(size_type n, const value_type& v);
// list(size_type n, const value_type& v, const allocator_type& a);
// template <class InputIterator>
//     list(InputIterator first, InputIterator last);
// template <class InputIterator>
//     list(InputIterator first, InputIterator last, const allocator_type& a);
// template<container-compatible-range<T> R>
//     list(from_range_t, R&& rg, const Allocator& = Allocator()); // C++23
// list(const list& x);
// list(const list& x, const allocator_type& a);
//
// list& operator=(const list& x);
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

#include <list>

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
      std::list<T> c;
      c.push_front(*from);
    });

    // void push_back(const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*){
      std::list<T> c;
      c.push_back(*from);
    });

    // iterator insert(const_iterator p, const value_type& v);
    test_exception_safety_throwing_copy</*ThrowOn=*/1, Size>([](T* from, T*){
      std::list<T> c;
      c.insert(c.end(), *from);
    });
  }

  {
    constexpr int ThrowOn = 3;
    constexpr int Size = 5;
    using T = ThrowingCopy<ThrowOn>;
    using C = std::list<T>;
    using Alloc = std::allocator<T>;

    // list(size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*){
      std::list<T> c(Size, *from);
      (void)c;
    });

    // list(size_type n, const value_type& v, const allocator_type& a);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*){
      std::list<T> c(Size, *from, Alloc());
      (void)c;
    });

    // template <class InputIterator>
    //     list(InputIterator first, InputIterator last);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to){
      std::list<T> c(from, to);
      (void)c;
    });

    // template <class InputIterator>
    //     list(InputIterator first, InputIterator last, const allocator_type& a);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to){
      std::list<T> c(from, to, Alloc());
      (void)c;
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //     list(from_range_t, R&& rg, const Allocator& = Allocator()); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to){
      {
        std::list<T> c(std::from_range, std::ranges::subrange(from, to));
        (void)c;
      }

      {
        std::list<T> c(std::from_range, std::ranges::subrange(from, to), Alloc());
        (void)c;
      }
    });
#endif

    // list(const list& x);
    test_exception_safety_throwing_copy_container<C, ThrowOn, Size>([](C&& in) {
      std::list<T> c(in);
      (void)c;
    });

    // list(const list& x, const allocator_type& a);
    test_exception_safety_throwing_copy_container<C, ThrowOn, Size>([](C&& in) {
      std::list<T> c(in, Alloc());
      (void)c;
    });

    // list& operator=(const list& x);
    test_exception_safety_throwing_copy_container<C, ThrowOn, Size>([](C&& in) {
      std::list<T> c;
      c = in;
    });

    // template <class InputIterator>
    //     void assign(InputIterator first, InputIterator last);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::list<T> c;
      c.assign(from, to);
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //     void assign_range(R&& rg); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::list<T> c;
      c.assign_range(std::ranges::subrange(from, to));
    });
#endif

    // void assign(size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::list<T> c;
      c.assign(Size, *from);
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //   void prepend_range(R&& rg); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::list<T> c;
      c.prepend_range(std::ranges::subrange(from, to));
    });

    // template<container-compatible-range<T> R>
    //   void append_range(R&& rg); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::list<T> c;
      c.append_range(std::ranges::subrange(from, to));
    });
#endif

    // iterator insert(const_iterator p, size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::list<T> c;
      c.insert(c.end(), Size, *from);
    });

    // template <class InputIterator>
    //     iterator insert(const_iterator p,
    //                     InputIterator first, InputIterator last);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::list<T> c;
      c.insert(c.end(), from, to);
    });

#if TEST_STD_VER >= 23
    // template<container-compatible-range<T> R>
    //     iterator insert_range(const_iterator position, R&& rg); // C++23
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T* to) {
      std::list<T> c;
      c.insert_range(c.end(), std::ranges::subrange(from, to));
    });
#endif

    // void resize(size_type n, const value_type& v);
    test_exception_safety_throwing_copy<ThrowOn, Size>([](T* from, T*) {
      std::list<T> c;
      c.resize(Size, *from);
    });
  }

  return 0;
}
