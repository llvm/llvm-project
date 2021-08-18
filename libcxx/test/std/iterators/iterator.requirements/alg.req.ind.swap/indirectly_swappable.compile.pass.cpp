//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class I1, class I2>
// concept indirectly_swappable;

#include <iterator>

#include "test_macros.h"

template<class T, class ValueType = T>
struct PointerTo {
  using value_type = ValueType;
  T& operator*() const;
};

static_assert(std::indirectly_swappable<PointerTo<int>>);
static_assert(std::indirectly_swappable<PointerTo<int>, PointerTo<int>>);

struct B;

struct A {
  friend void iter_swap(const PointerTo<A>&, const PointerTo<A>&);
};

// Is indirectly swappable.
struct B {
  friend void iter_swap(const PointerTo<B>&, const PointerTo<B>&);
  friend void iter_swap(const PointerTo<A>&, const PointerTo<B>&);
  friend void iter_swap(const PointerTo<B>&, const PointerTo<A>&);
};

// Valid except ranges::iter_swap(i2, i1).
struct C {
  friend void iter_swap(const PointerTo<C>&, const PointerTo<C>&);
  friend void iter_swap(const PointerTo<A>&, const PointerTo<C>&);
  friend void iter_swap(const PointerTo<C>&, const PointerTo<A>&) = delete;
};

// Valid except ranges::iter_swap(i1, i2).
struct D {
  friend void iter_swap(const PointerTo<D>&, const PointerTo<D>&);
  friend void iter_swap(const PointerTo<A>&, const PointerTo<D>&) = delete;
  friend void iter_swap(const PointerTo<D>&, const PointerTo<A>&);
};

// Valid except ranges::iter_swap(i2, i2).
struct E {
  E operator=(const E&) = delete;
  friend void iter_swap(const PointerTo<E>&, const PointerTo<E>&) = delete;
  friend void iter_swap(const PointerTo<A>&, const PointerTo<E>&);
  friend void iter_swap(const PointerTo<E>&, const PointerTo<A>&);
};

struct F {
  friend void iter_swap(const PointerTo<F>&, const PointerTo<F>&) = delete;
};

// Valid except ranges::iter_swap(i1, i1).
struct G {
  friend void iter_swap(const PointerTo<G>&, const PointerTo<G>&);
  friend void iter_swap(const PointerTo<F>&, const PointerTo<G>&);
  friend void iter_swap(const PointerTo<G>&, const PointerTo<F>&);
};


static_assert( std::indirectly_swappable<PointerTo<A>, PointerTo<B>>);
static_assert(!std::indirectly_swappable<PointerTo<A>, PointerTo<C>>);
static_assert(!std::indirectly_swappable<PointerTo<A>, PointerTo<D>>);
static_assert(!std::indirectly_swappable<PointerTo<A>, PointerTo<E>>);
static_assert(!std::indirectly_swappable<PointerTo<A>, PointerTo<G>>);
