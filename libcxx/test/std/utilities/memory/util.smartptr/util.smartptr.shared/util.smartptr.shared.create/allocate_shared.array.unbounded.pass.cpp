//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// This test requires support for aligned allocation to test overaligned types.
// XFAIL: availability-aligned_allocation-missing

// <memory>

// shared_ptr

// template<class T, class A>
// shared_ptr<T> allocate_shared(const A& a, size_t N); // T is U[]
//
// template<class T, class A>
// shared_ptr<T> allocate_shared(const A& a, size_t N, const remove_extent_t<T>& u); // T is U[]

// Ignore error about requesting a large alignment not being ABI compatible with older AIX systems.
#ifdef _AIX
# pragma clang diagnostic ignored "-Waix-compat"
#endif

#include <cassert>
#include <concepts>
#include <cstdint> // std::uintptr_t
#include <memory>
#include <utility>

#include "min_allocator.h"
#include "operator_hijacker.h"
#include "test_macros.h"
#include "types.h"

template <class T, class ...Args>
concept CanAllocateShared = requires(Args&& ...args) {
  { std::allocate_shared<T>(std::forward<Args>(args)...) } -> std::same_as<std::shared_ptr<T>>;
};

int main(int, char**) {
  // Check behavior for a zero-sized array
  {
    // Without passing an initial value
    {
      using Array = int[];
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 0);
      assert(ptr != nullptr);
    }

    // Passing an initial value
    {
      using Array = int[];
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 0, 42);
      assert(ptr != nullptr);
    }
  }

  // Check behavior for a 1-sized array
  {
    // Without passing an initial value
    {
      using Array = int[];
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 1);
      assert(ptr != nullptr);
      assert(ptr[0] == 0);
    }

    // Passing an initial value
    {
      using Array = int[];
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 1, 42);
      assert(ptr != nullptr);
      assert(ptr[0] == 42);
    }
  }

  // Make sure we initialize elements correctly
  {
    // Without passing an initial value
    {
      using Array = int[];
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
      for (unsigned i = 0; i < 8; ++i) {
        assert(ptr[i] == 0);
      }
    }
    {
      using Array = int[][3];
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
      for (unsigned i = 0; i < 8; ++i) {
        assert(ptr[i][0] == 0);
        assert(ptr[i][1] == 0);
        assert(ptr[i][2] == 0);
      }
    }
    {
      using Array = int[][3][2];
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
      for (unsigned i = 0; i < 8; ++i) {
        assert(ptr[i][0][0] == 0);
        assert(ptr[i][0][1] == 0);
        assert(ptr[i][1][0] == 0);
        assert(ptr[i][1][1] == 0);
        assert(ptr[i][2][0] == 0);
        assert(ptr[i][2][1] == 0);
      }
    }

    // Passing an initial value
    {
      using Array = int[];
      int init = 42;
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
      for (unsigned i = 0; i < 8; ++i) {
        assert(ptr[i] == init);
      }
    }
    {
      using Array = int[][3];
      int init[3] = {42, 43, 44};
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
      for (unsigned i = 0; i < 8; ++i) {
        assert(ptr[i][0] == 42);
        assert(ptr[i][1] == 43);
        assert(ptr[i][2] == 44);
      }
    }
    {
      using Array = int[][3][2];
      int init[3][2] = {{31, 32}, {41, 42}, {51, 52}};
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
      for (unsigned i = 0; i < 8; ++i) {
        assert(ptr[i][0][0] == 31);
        assert(ptr[i][0][1] == 32);
        assert(ptr[i][1][0] == 41);
        assert(ptr[i][1][1] == 42);
        assert(ptr[i][2][0] == 51);
        assert(ptr[i][2][1] == 52);
      }
    }
  }

  // Make sure array elements are destroyed in reverse order
  {
    // Without passing an initial value
    {
      using Array = DestroyInReverseOrder[];
      DestroyInReverseOrder::reset();
      {
        std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
        assert(DestroyInReverseOrder::alive() == 8);
      }
      assert(DestroyInReverseOrder::alive() == 0);
    }
    {
      using Array = DestroyInReverseOrder[][3];
      DestroyInReverseOrder::reset();
      {
        std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
        assert(DestroyInReverseOrder::alive() == 8 * 3);
      }
      assert(DestroyInReverseOrder::alive() == 0);
    }
    {
      using Array = DestroyInReverseOrder[][3][2];
      DestroyInReverseOrder::reset();
      {
        std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
        assert(DestroyInReverseOrder::alive() == 8 * 3 * 2);
      }
      assert(DestroyInReverseOrder::alive() == 0);
    }

    // Passing an initial value
    {
      using Array = DestroyInReverseOrder[];
      int count = 0;
      DestroyInReverseOrder init(&count);
      int init_count = 1;
      {
        std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
        assert(count == 8 + init_count);
      }
      assert(count == init_count);
    }
    {
      using Array = DestroyInReverseOrder[][3];
      int count = 0;
      DestroyInReverseOrder init[3] = {&count, &count, &count};
      int init_count = 3;
      {
        std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
        assert(count == 8 * 3 + init_count);
      }
      assert(count == init_count);
    }
    {
      using Array = DestroyInReverseOrder[][3][2];
      int count = 0;
      DestroyInReverseOrder init[3][2] = {{&count, &count}, {&count, &count}, {&count, &count}};
      int init_count = 3 * 2;
      {
        std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
        assert(count == 8 * 3 * 2 + init_count);
      }
      assert(count == init_count);
    }
  }

  // Count the number of copies being made
  {
    // Without passing an initial value
    {
      using Array = CountCopies[];
      CountCopies::reset();
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
      assert(CountCopies::copies() == 0);
    }
    {
      using Array = CountCopies[][3];
      CountCopies::reset();
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
      assert(CountCopies::copies() == 0);
    }
    {
      using Array = CountCopies[][3][2];
      CountCopies::reset();
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
      assert(CountCopies::copies() == 0);
    }

    // Passing an initial value
    {
      using Array = CountCopies[];
      int copies = 0;
      CountCopies init(&copies);
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
      assert(copies == 8);
    }
    {
      using Array = CountCopies[][3];
      int copies = 0;
      CountCopies init[3] = {&copies, &copies, &copies};
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
      assert(copies == 8 * 3);
    }
    {
      using Array = CountCopies[][3][2];
      int copies = 0;
      CountCopies init[3][2] = {{&copies, &copies}, {&copies, &copies}, {&copies, &copies}};
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
      assert(copies == 8 * 3 * 2);
    }
  }

  // Make sure array elements are aligned properly when the array contains an overaligned type.
  //
  // Here, we don't need to test both the with-initial-value and without-initial-value code paths,
  // since we're just checking the alignment and both are going to use the same code path unless
  // the implementation is completely crazy.
  {
    auto check_alignment = []<class T> {
      {
        using Array = T[];
        std::shared_ptr ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
        for (int i = 0; i < 8; ++i) {
          T* p = std::addressof(ptr[i]);
          assert(reinterpret_cast<std::uintptr_t>(p) % alignof(T) == 0);
        }
      }
      {
        using Array = T[][3];
        std::shared_ptr ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 3; ++j) {
            T* p = std::addressof(ptr[i][j]);
            assert(reinterpret_cast<std::uintptr_t>(p) % alignof(T) == 0);
          }
        }
      }
      {
        using Array = T[][3][2];
        std::shared_ptr ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 2; ++k) {
              T* p = std::addressof(ptr[i][j][k]);
              assert(reinterpret_cast<std::uintptr_t>(p) % alignof(T) == 0);
            }
          }
        }
      }
    };

    struct Empty { };
    check_alignment.operator()<Empty>();
    check_alignment.operator()<OverAligned>();
    check_alignment.operator()<MaxAligned>();

    // test non corner cases as well while we're at it
    struct Foo { int i; char c; };
    check_alignment.operator()<int>();
    check_alignment.operator()<Foo>();
  }

  // Make sure that we destroy all the elements constructed so far when an exception
  // is thrown. Also make sure that we do it in reverse order of construction.
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    struct Sentinel : ThrowOnConstruction, DestroyInReverseOrder { };

    // Without passing an initial value
    {
      using Array = Sentinel[];
      for (int i = 0; i < 8; ++i) {
        ThrowOnConstruction::throw_after(i);
        DestroyInReverseOrder::reset();
        try {
          std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
          assert(false);
        } catch (ThrowOnConstruction::exception const&) {
          assert(DestroyInReverseOrder::alive() == 0);
        }
      }
    }
    {
      using Array = Sentinel[][3];
      for (int i = 0; i < 8 * 3; ++i) {
        ThrowOnConstruction::throw_after(i);
        DestroyInReverseOrder::reset();
        try {
          std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
          assert(false);
        } catch (ThrowOnConstruction::exception const&) {
          assert(DestroyInReverseOrder::alive() == 0);
        }
      }
    }
    {
      using Array = Sentinel[][3][2];
      for (int i = 0; i < 8 * 3 * 2; ++i) {
        ThrowOnConstruction::throw_after(i);
        DestroyInReverseOrder::reset();
        try {
          std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
          assert(false);
        } catch (ThrowOnConstruction::exception const&) {
          assert(DestroyInReverseOrder::alive() == 0);
        }
      }
    }

    // Passing an initial value
    {
      using Array = Sentinel[];
      for (int i = 0; i < 8; ++i) {
        DestroyInReverseOrder::reset();
        ThrowOnConstruction::reset();
        Sentinel init;
        ThrowOnConstruction::throw_after(i);
        try {
          std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
          assert(false);
        } catch (ThrowOnConstruction::exception const&) {
          assert(DestroyInReverseOrder::alive() == 1);
        }
      }
    }
    {
      using Array = Sentinel[][3];
      for (int i = 0; i < 8 * 3; ++i) {
        DestroyInReverseOrder::reset();
        ThrowOnConstruction::reset();
        Sentinel init[3] = {};
        ThrowOnConstruction::throw_after(i);
        try {
          std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
          assert(false);
        } catch (ThrowOnConstruction::exception const&) {
          assert(DestroyInReverseOrder::alive() == 3);
        }
      }
    }
    {
      using Array = Sentinel[][3][2];
      for (int i = 0; i < 8 * 3 * 2; ++i) {
        DestroyInReverseOrder::reset();
        ThrowOnConstruction::reset();
        Sentinel init[3][2] = {};
        ThrowOnConstruction::throw_after(i);
        try {
          std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8, init);
          assert(false);
        } catch (ThrowOnConstruction::exception const&) {
          assert(DestroyInReverseOrder::alive() == 3 * 2);
        }
      }
    }
  }
#endif // TEST_HAS_NO_EXCEPTIONS

  // Test with another allocator that's not std::allocator
  {
    // Without passing an initial value
    {
      using Array = int[][3];
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(min_allocator<Array>(), 8);
      for (unsigned i = 0; i < 8; ++i) {
        assert(ptr[i][0] == 0);
        assert(ptr[i][1] == 0);
        assert(ptr[i][2] == 0);
      }
    }

    // Passing an initial value
    {
      using Array = int[][3];
      int init[3] = {42, 43, 44};
      std::shared_ptr<Array> ptr = std::allocate_shared<Array>(min_allocator<Array>(), 8, init);
      for (unsigned i = 0; i < 8; ++i) {
        assert(ptr[i][0] == 42);
        assert(ptr[i][1] == 43);
        assert(ptr[i][2] == 44);
      }
    }
  }

  // Make sure the version without an initialization argument works even for non-movable types
  {
    using Array = NonMovable[][3];
    std::shared_ptr<Array> ptr = std::allocate_shared<Array>(std::allocator<Array>(), 8);
    (void)ptr;
  }

  // Make sure std::allocate_shared handles badly-behaved types properly
  {
    using Array = operator_hijacker[];
    std::shared_ptr<Array> p1 = std::allocate_shared<Array>(std::allocator<Array>(), 3);
    std::shared_ptr<Array> p2 = std::allocate_shared<Array>(std::allocator<Array>(), 3, operator_hijacker());
    assert(p1 != nullptr);
    assert(p2 != nullptr);
  }

  // Check that we SFINAE-away for invalid arguments
  {
    struct T { };
    static_assert( CanAllocateShared<T[], std::allocator<T[]>, std::size_t>);
    static_assert( CanAllocateShared<T[], std::allocator<T[]>, std::size_t, T>);
    static_assert(!CanAllocateShared<T[], std::allocator<T[]>, std::size_t, T, int>); // too many arguments
    static_assert(!CanAllocateShared<T[], std::allocator<T[]>, std::size_t, int>); // T not constructible from int
  }

  return 0;
}
