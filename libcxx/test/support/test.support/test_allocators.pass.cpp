//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Makes sure that test allocators in "min_allocator.h" and "test_allocator.h" properly support
// heterogeneous construction and comparison.

#include <cassert>
#include <memory>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

#if TEST_STD_VER >= 11
template <class A, class U>
struct rebind_alloc {
  using type = typename std::allocator_traits<A>::template rebind_alloc<U>;
};
#else
template <class A, class U>
struct rebind_alloc {
  typedef typename std::allocator_traits<A>::template rebind_alloc<U>::other type;
};
#endif

TEST_CONSTEXPR_CXX20 bool test() {
  {
    ASSERT_SAME_TYPE(rebind_alloc<min_allocator<int>, char>::type, min_allocator<char>);
    min_allocator<int> a1;
    min_allocator<char> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<complete_type_allocator<int>, char>::type, complete_type_allocator<char>);
    complete_type_allocator<int> a1;
    complete_type_allocator<char> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<explicit_allocator<int>, char>::type, explicit_allocator<char>);
    explicit_allocator<int> a1;
    explicit_allocator<char> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<unaligned_allocator<unsigned char>, char>::type, unaligned_allocator<char>);
    unaligned_allocator<unsigned char> a1;
    unaligned_allocator<char> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<safe_allocator<int>, char>::type, safe_allocator<char>);
    safe_allocator<int> a1;
    safe_allocator<char> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<tiny_size_allocator<128, int>, char>::type, tiny_size_allocator<128, char>);
    tiny_size_allocator<128, int> a1;
    tiny_size_allocator<128, char> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<test_allocator<int>, char>::type, test_allocator<char>);
    test_allocator<int> a1(17);
    test_allocator<char> a2(a1);
    test_allocator<int> a3(29);
    test_allocator<char> a4(a3);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
    assert(!(a1 == a3));
    assert(!(a1 == a4));
    assert(a1 != a3);
    assert(a1 != a4);
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<other_allocator<int>, char>::type, other_allocator<char>);
    other_allocator<int> a1(17);
    other_allocator<char> a2(a1);
    other_allocator<int> a3(29);
    other_allocator<char> a4(a3);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
    assert(!(a1 == a3));
    assert(!(a1 == a4));
    assert(a1 != a3);
    assert(a1 != a4);
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<TaggingAllocator<int>, Tag_X>::type, TaggingAllocator<Tag_X>);
    TaggingAllocator<int> a1;
    TaggingAllocator<Tag_X> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<limited_allocator<int, 42>, char>::type, limited_allocator<char, 42>);
    limited_allocator<int, 42> a1;
    limited_allocator<char, 42> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  // constexpr-unfriendly allocators

  {
    ASSERT_SAME_TYPE(rebind_alloc<bare_allocator<int>, char>::type, bare_allocator<char>);
    bare_allocator<int> a1;
    bare_allocator<char> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<no_default_allocator<int>, char>::type, no_default_allocator<char>);
    no_default_allocator<int> a1 = no_default_allocator<int>::create();
    no_default_allocator<char> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<malloc_allocator<int>, char>::type, malloc_allocator<char>);
    malloc_allocator_base::disable_default_constructor = false;
    malloc_allocator<int> a1;
    malloc_allocator_base::disable_default_constructor = true;
    malloc_allocator<char> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<cpp03_allocator<int>, char>::type, cpp03_allocator<char>);
    cpp03_allocator<int> a1;
    cpp03_allocator<char> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<cpp03_overload_allocator<int>, char>::type, cpp03_overload_allocator<char>);
    cpp03_overload_allocator<int> a1;
    cpp03_overload_allocator<char> a2(a1);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
  }
  {
    ASSERT_SAME_TYPE(rebind_alloc<SocccAllocator<int>, char>::type, SocccAllocator<char>);
    SocccAllocator<int> a1(17);
    SocccAllocator<char> a2(a1);
    SocccAllocator<int> a3(29);
    SocccAllocator<char> a4(a3);

    assert(a1 == a1);
    assert(a1 == a2);
    assert(!(a1 != a1));
    assert(!(a1 != a2));
    assert(a1 == a3);
    assert(a1 == a4);
    assert(!(a1 != a3));
    assert(!(a1 != a4));
  }
  return 0;
}
