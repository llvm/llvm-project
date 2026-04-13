//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <tuple>

// See https://llvm.org/PR20855.

#include <functional>
#include <tuple>
#include <string>
#include <cassert>

template <class Tp>
struct ConvertsTo {
  using RawTp = typename std::remove_cv< typename std::remove_reference<Tp>::type>::type;

  operator Tp() const {
    return static_cast<Tp>(value);
  }

  mutable RawTp value;
};

struct Base {};
struct Derived : Base {};

static_assert(std::is_constructible<int&, std::reference_wrapper<int>>::value, "");
static_assert(std::is_constructible<int const&, std::reference_wrapper<int>>::value, "");

template <class T> struct CannotDeduce {
 using type = T;
};

template <class ...Args>
void F(typename CannotDeduce<std::tuple<Args...>>::type const&) {}

void compile_tests() {
  {
    F<int, int const&>(std::make_tuple(42, 42));
  }
  {
    F<int, int const&>(std::make_tuple<const int&, const int&>(42, 42));
    std::tuple<int, int const&> t(std::make_tuple<const int&, const int&>(42, 42));
  }
  {
    auto fn = &F<int, std::string const&>;
    fn(std::tuple<int, std::string const&>(42, std::string("a")));
    fn(std::make_tuple(42, std::string("a")));
  }
  {
    Derived d;
    std::tuple<Base&, Base const&> t(d, d);
  }
  {
    ConvertsTo<int&> ct;
    std::tuple<int, int&> t(42, ct);
  }
}

void allocator_tests() {
    std::allocator<int> alloc;
    int x = 42;
    {
        std::tuple<int&> t(std::ref(x));
        assert(&std::get<0>(t) == &x);
        std::tuple<int&> t1(std::allocator_arg, alloc, std::ref(x));
        assert(&std::get<0>(t1) == &x);
    }
    {
        auto r = std::ref(x);
        auto const& cr = r;
        std::tuple<int&> t(r);
        assert(&std::get<0>(t) == &x);
        std::tuple<int&> t1(cr);
        assert(&std::get<0>(t1) == &x);
        std::tuple<int&> t2(std::allocator_arg, alloc, r);
        assert(&std::get<0>(t2) == &x);
        std::tuple<int&> t3(std::allocator_arg, alloc, cr);
        assert(&std::get<0>(t3) == &x);
    }
    {
        std::tuple<int const&> t(std::ref(x));
        assert(&std::get<0>(t) == &x);
        std::tuple<int const&> t2(std::cref(x));
        assert(&std::get<0>(t2) == &x);
        std::tuple<int const&> t3(std::allocator_arg, alloc, std::ref(x));
        assert(&std::get<0>(t3) == &x);
        std::tuple<int const&> t4(std::allocator_arg, alloc, std::cref(x));
        assert(&std::get<0>(t4) == &x);
    }
    {
        auto r = std::ref(x);
        auto cr = std::cref(x);
        std::tuple<int const&> t(r);
        assert(&std::get<0>(t) == &x);
        std::tuple<int const&> t2(cr);
        assert(&std::get<0>(t2) == &x);
        std::tuple<int const&> t3(std::allocator_arg, alloc, r);
        assert(&std::get<0>(t3) == &x);
        std::tuple<int const&> t4(std::allocator_arg, alloc, cr);
        assert(&std::get<0>(t4) == &x);
    }
}


int main(int, char**) {
  compile_tests();
  allocator_tests();

  return 0;
}
