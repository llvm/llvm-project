//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

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
#include <type_traits>

#include "test_macros.h"

template <typename... Ts>
struct VoidifyImpl {
  using type = void;
};

template <typename... Ts>
using Voidify = typename VoidifyImpl<Ts...>::type;

template <class T, class = void>
struct HasElementType : std::false_type {};

template <class T>
struct HasElementType<T, Voidify<typename std::pointer_traits<T>::element_type> > : std::true_type {};

template <class T, class = void>
struct HasPointerType : std::false_type {};

template <class T>
struct HasPointerType<T, Voidify<typename std::pointer_traits<T>::pointer> > : std::true_type {};

template <class T, class = void>
struct HasDifferenceType : std::false_type {};

template <class T>
struct HasDifferenceType<T, Voidify<typename std::pointer_traits<T>::difference_type> > : std::true_type {};

template <class T, class U, class = void>
struct HasRebind : std::false_type {};

template <class T, class U>
struct HasRebind<T, U, Voidify<typename std::pointer_traits<T>::template rebind<U> > > : std::true_type {};

template <class T, class = void>
struct HasPointerTo : std::false_type {};

template <class T>
struct HasPointerTo<
    T,
    Voidify<decltype(std::pointer_traits<T>::pointer_to(
        std::declval<typename std::add_lvalue_reference<typename std::pointer_traits<T>::element_type>::type>()))> >
    : std::true_type {};

struct Irrelevant;

struct NotAPtr {};

struct LongPtr {};

int global_int;

template <class T, class Arg>
struct TemplatedPtr;

struct PtrWithElementType {
  using element_type = int;
  template <class U>
#if TEST_STD_VER >= 11
  using rebind = TemplatedPtr<U, Irrelevant>;
#else
  struct rebind {
    using other = TemplatedPtr<U, Irrelevant>;
  };
#endif
  static TEST_CONSTEXPR_CXX14 PtrWithElementType pointer_to(element_type&) { return {&global_int}; }
  int* ptr;
};

template <class T, class Arg>
struct TemplatedPtr {
  template <class U, class = typename std::enable_if<std::is_same<long, U>::value>::type>
#if TEST_STD_VER >= 11
  using rebind = LongPtr;
#else
  struct rebind {
    using other = LongPtr;
  };
#endif
  static TEST_CONSTEXPR_CXX14 TemplatedPtr pointer_to(T&) { return {&global_int}; }

  T* ptr;
};

template <class T, class Arg>
struct TemplatedPtrWithElementType {
  using element_type = int;
#if TEST_STD_VER >= 11
  template <class U, class = typename std::enable_if<std::is_same<long, U>::value>::type>
  using rebind = LongPtr;
#else
  template <class U, class = typename std::enable_if<std::is_same<long, U>::value>::type>
  struct rebind {
    using other = LongPtr;
  };
#endif
  static TEST_CONSTEXPR_CXX14 TemplatedPtrWithElementType pointer_to(element_type&) { return {&global_int}; }

  element_type* ptr;
};

int main() {
  {
    using Ptr = NotAPtr;
    static_assert(!HasElementType<Ptr>::value, "");
    static_assert(!HasPointerType<Ptr>::value, "");
    static_assert(!HasDifferenceType<Ptr>::value, "");
    static_assert(!HasRebind<Ptr, long>::value, "");
    static_assert(!HasPointerTo<Ptr>::value, "");
  }

  {
    using Ptr = int*;

    static_assert(HasElementType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::element_type, int);

    static_assert(HasPointerType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::pointer, Ptr);

    static_assert(HasDifferenceType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::difference_type, ptrdiff_t);

    static_assert(HasRebind<Ptr, long>::value, "");
#if TEST_STD_VER >= 11
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>, long*);
#else
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>::other, long*);
#endif

    static_assert(HasPointerTo<Ptr>::value, "");
    int variable = 0;
    ASSERT_SAME_TYPE(decltype(std::pointer_traits<Ptr>::pointer_to(variable)), Ptr);
#if TEST_STD_VER >= 20
    static_assert(std::pointer_traits<Ptr>::pointer_to(variable) == &variable, "");
#endif
  }

  {
    using Ptr = const int*;

    static_assert(HasElementType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::element_type, const int);

    static_assert(HasPointerType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::pointer, Ptr);

    static_assert(HasDifferenceType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::difference_type, ptrdiff_t);

    static_assert(HasRebind<Ptr, long>::value, "");
#if TEST_STD_VER >= 11
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>, long*);
#else
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>::other, long*);
#endif

    static_assert(HasPointerTo<Ptr>::value, "");
    const int const_variable = 0;
    ASSERT_SAME_TYPE(decltype(std::pointer_traits<Ptr>::pointer_to(const_variable)), Ptr);
#if TEST_STD_VER >= 20
    static_assert(std::pointer_traits<Ptr>::pointer_to(const_variable) == &const_variable, "");
#endif
    int variable = 0;
    ASSERT_SAME_TYPE(decltype(std::pointer_traits<Ptr>::pointer_to(variable)), Ptr);
#if TEST_STD_VER >= 20
    static_assert(std::pointer_traits<Ptr>::pointer_to(variable) == &variable, "");
#endif
  }

  {
    using Ptr = PtrWithElementType;

    static_assert(HasElementType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::element_type, int);

    static_assert(HasPointerType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::pointer, Ptr);

    static_assert(HasDifferenceType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::difference_type, ptrdiff_t);

    static_assert(HasRebind<Ptr, long>::value, "");
#if TEST_STD_VER >= 11
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>, TemplatedPtr<long, Irrelevant>);
#else
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>::other, TemplatedPtr<long, Irrelevant>);
#endif

    static_assert(HasPointerTo<Ptr>::value, "");
    int ignored = 0;
    ASSERT_SAME_TYPE(decltype(std::pointer_traits<Ptr>::pointer_to(ignored)), Ptr);
#if TEST_STD_VER >= 20
    static_assert(std::pointer_traits<Ptr>::pointer_to(ignored).ptr == &global_int, "");
#endif
  }

  {
    using Ptr = TemplatedPtr<int, Irrelevant>;

    static_assert(HasElementType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::element_type, int);

    static_assert(HasPointerType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::pointer, Ptr);

    static_assert(HasDifferenceType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::difference_type, ptrdiff_t);

    static_assert(HasRebind<Ptr, long>::value, "");
    static_assert(HasRebind<Ptr, long long>::value, "");
#if TEST_STD_VER >= 11
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>, LongPtr);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long long>, TemplatedPtr<long long, Irrelevant>);
#else
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>::other, LongPtr);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long long>::other, TemplatedPtr<long long, Irrelevant>);
#endif

    static_assert(HasPointerTo<Ptr>::value, "");
    int ignored = 0;
    ASSERT_SAME_TYPE(decltype(std::pointer_traits<Ptr>::pointer_to(ignored)), Ptr);
#if TEST_STD_VER >= 20
    static_assert(std::pointer_traits<Ptr>::pointer_to(ignored).ptr == &global_int, "");
#endif
  }

  {
    using Ptr = TemplatedPtrWithElementType<Irrelevant, Irrelevant>;

    static_assert(HasElementType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::element_type, int);

    static_assert(HasPointerType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::pointer, Ptr);

    static_assert(HasDifferenceType<Ptr>::value, "");
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::difference_type, ptrdiff_t);

    static_assert(HasRebind<Ptr, long>::value, "");
    static_assert(HasRebind<Ptr, long long>::value, "");
#if TEST_STD_VER >= 11
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>, LongPtr);
    ASSERT_SAME_TYPE(
        typename std::pointer_traits<Ptr>::rebind<long long>, TemplatedPtrWithElementType<long long, Irrelevant>);
#else
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long>::other, LongPtr);
    ASSERT_SAME_TYPE(typename std::pointer_traits<Ptr>::rebind<long long>::other,
                     TemplatedPtrWithElementType<long long, Irrelevant>);
#endif

    static_assert(HasPointerTo<Ptr>::value, "");
    int ignored = 0;
    ASSERT_SAME_TYPE(decltype(std::pointer_traits<Ptr>::pointer_to(ignored)), Ptr);
#if TEST_STD_VER >= 20
    static_assert(std::pointer_traits<Ptr>::pointer_to(ignored).ptr == &global_int, "");
#endif
  }
}
