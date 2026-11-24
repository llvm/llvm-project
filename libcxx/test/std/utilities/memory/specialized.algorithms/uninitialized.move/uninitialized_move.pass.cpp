//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <memory>

// template <class InputIt, class ForwardIt>
// ForwardIt uninitialized_move(InputIt, InputIt, ForwardIt);

#include <memory>
#include <cstdlib>
#include <cassert>

#include "MoveOnly.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "../overload_compare_iterator.h"

struct Counted {
  static int count;
  static int constructed;
  static void reset() { count = constructed =  0; }
  explicit Counted(int&& x) : value(x) { x = 0; ++count; ++constructed; }
  Counted(Counted const&) { assert(false); }
  ~Counted() { assert(count > 0); --count; }
  friend void operator&(Counted) = delete;
  int value;
};
int Counted::count = 0;
int Counted::constructed = 0;

struct ThrowsCounted {
  static int count;
  static int constructed;
  static int throw_after;
  static void reset() { throw_after = count = constructed =  0; }
  explicit ThrowsCounted(int&& x) {
      ++constructed;
      if (throw_after > 0 && --throw_after == 0) {
          TEST_THROW(1);
      }
      ++count;
      x = 0;
  }
  ThrowsCounted(ThrowsCounted const&) { assert(false); }
  ~ThrowsCounted() { assert(count > 0); --count; }
  friend void operator&(ThrowsCounted) = delete;
};
int ThrowsCounted::count = 0;
int ThrowsCounted::constructed = 0;
int ThrowsCounted::throw_after = 0;

struct NoMoveNoCopy {
  constexpr explicit NoMoveNoCopy(int x) : value(x) {}
  NoMoveNoCopy(const NoMoveNoCopy&)   = delete;
  friend void operator&(NoMoveNoCopy) = delete;
  int value;
};

class PrvalueIterator {
public:
  using iterator_category = std::input_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using reference         = NoMoveNoCopy;
  using pointer           = void;
  using value_type        = NoMoveNoCopy;

  PrvalueIterator() = delete;
  constexpr explicit PrvalueIterator(const int* ptr) : ptr_(ptr) {}

  constexpr NoMoveNoCopy operator*() const { return NoMoveNoCopy(*ptr_); }

  constexpr PrvalueIterator& operator++() {
    ++ptr_;
    return *this;
  }

  friend constexpr bool operator==(PrvalueIterator a, PrvalueIterator b) { return a.ptr_ == b.ptr_; }
  friend constexpr bool operator!=(PrvalueIterator a, PrvalueIterator b) { return a.ptr_ != b.ptr_; }

private:
  const int* ptr_;
};

TEST_CONSTEXPR_CXX26 bool test() {
  const int n    = 3;
  MoveOnly in[n] = {1, 2, 3};
  std::allocator<MoveOnly> alloc;
  MoveOnly* out = alloc.allocate(n);

  MoveOnly* result = std::uninitialized_move(in, in + n, out);
  assert(result == out + n);
  for (int i = 0; i < n; ++i) {
    assert(in[i] == 0);
    assert(out[i] == i + 1);
  }

  std::destroy(out, out + n);
  alloc.deallocate(out, n);

  return true;
}

void test_ctor_throws()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    using It = forward_iterator<ThrowsCounted*>;
    const int N = 5;
    int values[N] = {1, 2, 3, 4, 5};
    alignas(ThrowsCounted) char pool[sizeof(ThrowsCounted)*N] = {};
    ThrowsCounted* p = (ThrowsCounted*)pool;
    try {
        ThrowsCounted::throw_after = 4;
        std::uninitialized_move(values, values + N, It(p));
        assert(false);
    } catch (...) {}
    assert(ThrowsCounted::count == 0);
    assert(ThrowsCounted::constructed == 4); // forth construction throws
    assert(values[0] == 0);
    assert(values[1] == 0);
    assert(values[2] == 0);
    assert(values[3] == 4);
    assert(values[4] == 5);
#endif
}

void test_counted()
{
    using It = cpp17_input_iterator<int*>;
    using FIt = forward_iterator<Counted*>;
    const int N = 5;
    int values[N] = {1, 2, 3, 4, 5};
    alignas(Counted) char pool[sizeof(Counted)*N] = {};
    Counted* p = (Counted*)pool;
    auto ret = std::uninitialized_move(It(values), It(values + 1), FIt(p));
    assert(ret == FIt(p +1));
    assert(Counted::constructed == 1);
    assert(Counted::count == 1);
    assert(p[0].value == 1);
    assert(values[0] == 0);
    ret = std::uninitialized_move(It(values+1), It(values+N), FIt(p+1));
    assert(ret == FIt(p + N));
    assert(Counted::count == 5);
    assert(Counted::constructed == 5);
    assert(p[1].value == 2);
    assert(p[2].value == 3);
    assert(p[3].value == 4);
    assert(p[4].value == 5);
    assert(values[1] == 0);
    assert(values[2] == 0);
    assert(values[3] == 0);
    assert(values[4] == 0);
    std::destroy(p, p+N);
    assert(Counted::count == 0);
}

TEST_CONSTEXPR_CXX26 bool test_copy_elision() {
  using It      = PrvalueIterator;
  using FIt     = forward_iterator<NoMoveNoCopy*>;
  const int N   = 5;
  int values[N] = {1, 2, 3, 4, 5};
  std::allocator<NoMoveNoCopy> alloc;
  NoMoveNoCopy* p = alloc.allocate(N);
  auto ret        = std::uninitialized_move(It(values), It(values + 1), FIt(p));
  assert(ret == FIt(p + 1));
  assert(p[0].value == 1);
  assert(values[0] == 1);
  ret = std::uninitialized_move(It(values + 1), It(values + N), FIt(p + 1));
  assert(p[1].value == 2);
  assert(p[2].value == 3);
  assert(p[3].value == 4);
  assert(p[4].value == 5);
  assert(values[1] == 2);
  assert(values[2] == 3);
  assert(values[3] == 4);
  assert(values[4] == 5);
  std::destroy(p, p + N);
  alloc.deallocate(p, N);

  return true;
}

int main(int, char**) {
    test_counted();
    test_ctor_throws();
    test_copy_elision();

    // Test with an iterator that overloads operator== and operator!= as the input and output iterators
    {
        using T = int;
        using Iterator = overload_compare_iterator<T*>;
        const int N = 5;

        // input
        {
            char pool[sizeof(T) * N] = {0};
            T* p = reinterpret_cast<T*>(pool);
            T array[N] = {1, 2, 3, 4, 5};
            std::uninitialized_move(Iterator(array), Iterator(array + N), p);
            for (int i = 0; i != N; ++i) {
                assert(array[i] == p[i]);
            }
        }

        // output
        {
            char pool[sizeof(T) * N] = {0};
            T* p = reinterpret_cast<T*>(pool);
            T array[N] = {1, 2, 3, 4, 5};
            std::uninitialized_move(array, array + N, Iterator(p));
            for (int i = 0; i != N; ++i) {
                assert(array[i] == p[i]);
            }
        }
    }

    test();
#if TEST_STD_VER >= 26
    static_assert(test());
    static_assert(test_copy_elision());
#endif

    return 0;
}
