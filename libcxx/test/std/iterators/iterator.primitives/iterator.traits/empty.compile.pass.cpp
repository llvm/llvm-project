//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// struct iterator_traits
// {
// };

#include <iterator>
#include <type_traits>

#include "test_macros.h"

template <class...>
using always_void = void;

#define HAS_XXX(member)                                                                                                \
  template <class T, class = void>                                                                                     \
  struct has_##member : std::false_type {};                                                                            \
  template <class T>                                                                                                   \
  struct has_##member<T, always_void<typename T::member> > : std::true_type {}

HAS_XXX(difference_type);
HAS_XXX(value_type);
HAS_XXX(pointer);
HAS_XXX(reference);
HAS_XXX(iterator_category);

struct A {};
struct NotAnIteratorEmpty {};

struct NotAnIteratorNoDifference {
  //     typedef int                       difference_type;
  typedef A value_type;
  typedef A* pointer;
  typedef A& reference;
  typedef std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoValue {
  typedef int difference_type;
  //     typedef A                         value_type;
  typedef A* pointer;
  typedef A& reference;
  typedef std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoPointer {
  typedef int difference_type;
  typedef A value_type;
  //     typedef A*                        pointer;
  typedef A& reference;
  typedef std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoReference {
  typedef int difference_type;
  typedef A value_type;
  typedef A* pointer;
  //    typedef A&                        reference;
  typedef std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoCategory {
  typedef int difference_type;
  typedef A value_type;
  typedef A* pointer;
  typedef A& reference;
  //     typedef std::forward_iterator_tag iterator_category;
};

void test() {
  {
    typedef std::iterator_traits<NotAnIteratorEmpty> T;
    static_assert(!has_difference_type<T>::value, "");
    static_assert(!has_value_type<T>::value, "");
    static_assert(!has_pointer<T>::value, "");
    static_assert(!has_reference<T>::value, "");
    static_assert(!has_iterator_category<T>::value, "");
  }

  {
    typedef std::iterator_traits<NotAnIteratorNoDifference> T;
    static_assert(!has_difference_type<T>::value, "");
    static_assert(!has_value_type<T>::value, "");
    static_assert(!has_pointer<T>::value, "");
    static_assert(!has_reference<T>::value, "");
    static_assert(!has_iterator_category<T>::value, "");
  }

  {
    typedef std::iterator_traits<NotAnIteratorNoValue> T;
    static_assert(!has_difference_type<T>::value, "");
    static_assert(!has_value_type<T>::value, "");
    static_assert(!has_pointer<T>::value, "");
    static_assert(!has_reference<T>::value, "");
    static_assert(!has_iterator_category<T>::value, "");
  }
#if TEST_STD_VER <= 17 || !defined(__cpp_lib_concepts)
  {
    typedef std::iterator_traits<NotAnIteratorNoPointer> T;
    static_assert(!has_difference_type<T>::value, "");
    static_assert(!has_value_type<T>::value, "");
    static_assert(!has_pointer<T>::value, "");
    static_assert(!has_reference<T>::value, "");
    static_assert(!has_iterator_category<T>::value, "");
  }
#endif // TEST_STD_VER <= 17 || !defined(__cpp_lib_concepts)
  {
    typedef std::iterator_traits<NotAnIteratorNoReference> T;
    static_assert(!has_difference_type<T>::value, "");
    static_assert(!has_value_type<T>::value, "");
    static_assert(!has_pointer<T>::value, "");
    static_assert(!has_reference<T>::value, "");
    static_assert(!has_iterator_category<T>::value, "");
  }

  {
    typedef std::iterator_traits<NotAnIteratorNoCategory> T;
    static_assert(!has_difference_type<T>::value, "");
    static_assert(!has_value_type<T>::value, "");
    static_assert(!has_pointer<T>::value, "");
    static_assert(!has_reference<T>::value, "");
    static_assert(!has_iterator_category<T>::value, "");
  }
}
