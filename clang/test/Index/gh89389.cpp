// RUN: c-index-test -test-load-source all %s -std=gnu++20 -fno-delayed-template-parsing

namespace test18 {
template<typename T>
concept False = false;

template <typename T> struct Foo { T t; };

template<typename T> requires False<T>
Foo(T) -> Foo<int>;

template <typename U>
using Bar = Foo<U>;

Bar s = {1};
} // namespace test18
