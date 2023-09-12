//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class Ptr>
// struct pointer_traits
// {
//     <details>
// };
//
// template <class T>
// struct pointer_traits<T*>
// {
//     using pointer = T*;
//     using element_type = T;
//     using difference_type = ptrdiff_t;
//     template <class U> using rebind = U*;
//     static constexpr pointer pointer_to(<details>) noexcept;
//     ...
// };

#include <memory>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <class T, class = void>
struct HasElementType : std::false_type {};

template <class T>
struct HasElementType<T, std::void_t<typename std::pointer_traits<T>::element_type>> : std::true_type {};

template <class T, class = void>
struct HasPointerType : std::false_type {};

template <class T>
struct HasPointerType<T, std::void_t<typename std::pointer_traits<T>::pointer>> : std::true_type {};

template <class T, class = void>
struct HasDifferenceType : std::false_type {};

template <class T>
struct HasDifferenceType<T, std::void_t<typename std::pointer_traits<T>::difference_type>> : std::true_type {};

template <class T, class U, class = void>
struct HasRebind : std::false_type {};

template <class T, class U>
struct HasRebind<T, U, std::void_t<typename std::pointer_traits<T>::template rebind<U>>> : std::true_type {};

template <class T, class = void>
struct HasPointerTo : std::false_type {};

template <class T>
struct HasPointerTo<T,
                    std::void_t<decltype(std::pointer_traits<T>::pointer_to(
                        std::declval<std::add_lvalue_reference_t<typename std::pointer_traits<T>::element_type>>()))>>
    : std::true_type {};

struct Irrelevant;

struct NotAPtr {};

struct LongPtr;

int global_int;

template <class T, class Arg>
struct TemplatedPtr;

struct PtrWithElementType {
  using element_type = int;
  template <typename U>
  using rebind = TemplatedPtr<U, Irrelevant>;
  static constexpr auto pointer_to(element_type&) { return PtrWithElementType{&global_int}; }

  int* ptr;
};

template <class T, class Arg>
struct TemplatedPtr {
  template <typename U, typename = std::enable_if_t<std::is_same_v<long, U>>>
  using rebind = LongPtr;
  static constexpr auto pointer_to(T&) { return TemplatedPtr{&global_int}; }

  T* ptr;
};

template <class T, class Arg>
struct TemplatedPtrWithElementType {
  using element_type = int;
  template <typename U, typename = std::enable_if_t<std::is_same_v<long, U>>>
  using rebind = LongPtr;
  static constexpr auto pointer_to(element_type&) { return TemplatedPtrWithElementType{&global_int}; }

  element_type* ptr;
};

constexpr bool test() {
  {
    using Ptr = NotAPtr;
    assert(!HasElementType<Ptr>::value);
    assert(!HasPointerType<Ptr>::value);
    assert(!HasDifferenceType<Ptr>::value);
    assert((!HasRebind<Ptr, long>::value));
    assert(!HasPointerTo<Ptr>::value);
  }

  {
    using Ptr = int*;

    assert(HasElementType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::element_type, int);

    assert(HasPointerType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::pointer, Ptr);

    assert(HasDifferenceType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::difference_type, ptrdiff_t);

    assert((HasRebind<Ptr, long>::value));
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>, long*);

    assert(HasPointerTo<Ptr>::value);
    int variable;
    ASSERT_SAME_TYPE(decltype(std::pointer_traits<Ptr>::pointer_to(variable)), int*);
    assert(std::pointer_traits<Ptr>::pointer_to(variable) == &variable);
  }

  {
    using Ptr = const int*;

    assert(HasElementType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::element_type, const int);

    assert(HasPointerType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::pointer, Ptr);

    assert(HasDifferenceType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::difference_type, ptrdiff_t);

    assert((HasRebind<Ptr, long>::value));
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>, long*);

    assert(HasPointerTo<Ptr>::value);
    const int const_variable = 0;
    ASSERT_SAME_TYPE(decltype(std::pointer_traits<Ptr>::pointer_to(const_variable)), const int*);
    assert(std::pointer_traits<Ptr>::pointer_to(const_variable) == &const_variable);
    int variable = 0;
    ASSERT_SAME_TYPE(decltype(std::pointer_traits<Ptr>::pointer_to(variable)), const int*);
    assert(std::pointer_traits<Ptr>::pointer_to(variable) == &variable);
  }

  {
    using Ptr = PtrWithElementType;

    assert(HasElementType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::element_type, int);

    assert(HasPointerType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::pointer, Ptr);

    assert(HasDifferenceType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::difference_type, ptrdiff_t);

    // TODO: Consider supporting SFINAE testing of std::pointer_traits<Ptr>.
    assert((HasRebind<Ptr, long>::value));
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>, TemplatedPtr<long, Irrelevant>);

    assert(HasPointerTo<Ptr>::value);
    int ignored;
    assert(std::pointer_traits<Ptr>::pointer_to(ignored).ptr == &global_int);
  }

  {
    using Ptr = TemplatedPtr<int, Irrelevant>;

    assert(HasElementType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::element_type, int);

    assert(HasPointerType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::pointer, Ptr);

    assert(HasDifferenceType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::difference_type, ptrdiff_t);

    assert((HasRebind<Ptr, long>::value));
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>, LongPtr);
    assert((HasRebind<Ptr, long long>::value));
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long long>, TemplatedPtr<long long, Irrelevant>);

    assert(HasPointerTo<Ptr>::value);
    int ignored;
    assert(std::pointer_traits<Ptr>::pointer_to(ignored).ptr == &global_int);
  }

  {
    using Ptr = TemplatedPtrWithElementType<Irrelevant, Irrelevant>;

    assert(HasElementType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::element_type, int);

    assert(HasPointerType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::pointer, Ptr);

    assert(HasDifferenceType<Ptr>::value);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::difference_type, ptrdiff_t);

    assert((HasRebind<Ptr, long>::value));
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>, LongPtr);
    assert((HasRebind<Ptr, long long>::value));
    ASSERT_SAME_TYPE(
        typename std::pointer_traits<Ptr>::rebind<long long>, TemplatedPtrWithElementType<long long, Irrelevant>);

    assert(HasPointerTo<Ptr>::value);
    int ignored;
    assert(std::pointer_traits<Ptr>::pointer_to(ignored).ptr == &global_int);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
