//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// template<class U, class V> pair& operator=(tuple<U, V>&& p);

#include <utility>
#include <tuple>
#include <array>
#include <memory>
#include <cassert>

// Clang warns about missing braces when initializing std::array.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wmissing-braces"
#endif

struct CountingType {
  static int constructed;
  static int copy_constructed;
  static int move_constructed;
  static int assigned;
  static int copy_assigned;
  static int move_assigned;
  static void reset() {
      constructed = copy_constructed = move_constructed = 0;
      assigned = copy_assigned = move_assigned = 0;
  }
  CountingType() : value(0) { ++constructed; }
  CountingType(int v) : value(v) { ++constructed; }
  CountingType(CountingType const& o) : value(o.value) { ++constructed; ++copy_constructed; }
  CountingType(CountingType&& o) : value(o.value) { ++constructed; ++move_constructed; o.value = -1;}

  CountingType& operator=(CountingType const& o) {
      ++assigned;
      ++copy_assigned;
      value = o.value;
      return *this;
  }
  CountingType& operator=(CountingType&& o) {
      ++assigned;
      ++move_assigned;
      value = o.value;
      o.value = -1;
      return *this;
  }
  int value;
};
int CountingType::constructed;
int CountingType::copy_constructed;
int CountingType::move_constructed;
int CountingType::assigned;
int CountingType::copy_assigned;
int CountingType::move_assigned;

int main()
{
    using C = CountingType;
    {
       using P = std::pair<int, C>;
       using T = std::tuple<int, C>;
       T t(42, C{42});
       P p(101, C{101});
       C::reset();
       p = t;
       assert(C::constructed == 0);
       assert(C::assigned == 1);
       assert(C::copy_assigned == 1);
       assert(C::move_assigned == 0);
       assert(p.first == 42);
       assert(p.second.value == 42);
    }
    {
       using P = std::pair<int, C>;
       using T = std::tuple<int, C>;
       T t(42, -42);
       P p(101, 101);
       C::reset();
       p = std::move(t);
       assert(C::constructed == 0);
       assert(C::assigned == 1);
       assert(C::copy_assigned == 0);
       assert(C::move_assigned == 1);
       assert(p.first == 42);
       assert(p.second.value == -42);
    }
    {
       using P = std::pair<C, C>;
       using T = std::array<C, 2>;
       T t = {42, -42};
       P p{101, 101};
       C::reset();
       p = t;
       assert(C::constructed == 0);
       assert(C::assigned == 2);
       assert(C::copy_assigned == 2);
       assert(C::move_assigned == 0);
       assert(p.first.value == 42);
       assert(p.second.value == -42);
    }
    {
       using P = std::pair<C, C>;
       using T = std::array<C, 2>;
       T t = {42, -42};
       P p{101, 101};
       C::reset();
       p = t;
       assert(C::constructed == 0);
       assert(C::assigned == 2);
       assert(C::copy_assigned == 2);
       assert(C::move_assigned == 0);
       assert(p.first.value == 42);
       assert(p.second.value == -42);
    }
    {
       using P = std::pair<C, C>;
       using T = std::array<C, 2>;
       T t = {42, -42};
       P p{101, 101};
       C::reset();
       p = std::move(t);
       assert(C::constructed == 0);
       assert(C::assigned == 2);
       assert(C::copy_assigned == 0);
       assert(C::move_assigned == 2);
       assert(p.first.value == 42);
       assert(p.second.value == -42);
    }
}
