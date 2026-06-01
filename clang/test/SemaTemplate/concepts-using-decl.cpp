// RUN: %clang_cc1 -std=c++20 -verify %s

namespace static_methods {
template<class> concept False = false;

struct Base {
    static void foo(auto);
};
struct Derived : public Base {
    using Base::foo;
    static void foo(False auto);
};
void func() {
    Derived::foo(42);
}
} // namespace static_methods

namespace constrained_members {
template <unsigned n> struct Opaque {};
template <unsigned n> void expect(Opaque<n> _) {}

struct Empty{};
constexpr int EmptySize = sizeof(Empty);

template<typename T> concept IsEmpty = sizeof(T) == EmptySize;

namespace base_members_not_hidden {
struct base {
  template <typename T>
  Opaque<0> foo() { return Opaque<0>(); };
};

struct bar1 : public base {
  using base::foo;
  template <typename T> requires IsEmpty<T> 
  Opaque<1> foo() { return Opaque<1>(); };
};

struct bar2 : public base {
  using base::foo;
  template <IsEmpty T>
  Opaque<1> foo() { return Opaque<1>(); };
};

struct bar3 : public base {
  using base::foo;
  template <typename T>
  Opaque<1> foo() requires IsEmpty<T> { return Opaque<1>(); };
};

void func() {
  expect<0>(base{}.foo<Empty>());
  expect<0>(base{}.foo<int>());
  expect<1>(bar1{}.foo<Empty>());
  expect<0>(bar1{}.foo<int>());
  expect<1>(bar2{}.foo<Empty>());
  expect<0>(bar2{}.foo<int>());
  expect<1>(bar3{}.foo<Empty>());
  expect<0>(bar3{}.foo<int>());
}
}
namespace base_members_hidden {
struct base1 {
  template <typename T> requires IsEmpty<T>
  Opaque<0> foo() { return Opaque<0>(); }; // expected-note {{candidate function}}
};
struct bar1 : public base1 {
  using base1::foo;
  template <typename T> requires IsEmpty<T>
  Opaque<1> foo() { return Opaque<1>(); };
};
struct base2 {
  template <IsEmpty T>
  Opaque<0> foo() { return Opaque<0>(); };
};
struct bar2 : public base2 {
  using base2::foo;
  template <IsEmpty T>
  Opaque<1> foo() { return Opaque<1>(); };
};
struct baz : public base1 {
  using base1::foo;
  template <typename T> requires IsEmpty<T> && IsEmpty<T>
  Opaque<1> foo() { return Opaque<1>(); };  // expected-note {{candidate function}}
};
void func() { 
  expect<0>(base1{}.foo<Empty>());
  expect<1>(bar1{}.foo<Empty>());
  expect<0>(base2{}.foo<Empty>());
  expect<1>(bar2{}.foo<Empty>());
  baz{}.foo<Empty>(); // expected-error {{call to member function 'foo' is ambiguous}}
}
} // namespace base_members_hidden

namespace same_contraint_at_different_place {
struct base {
  template <IsEmpty T>
  void foo1() {}; // expected-note 2 {{candidate function}}
  template <typename T> requires IsEmpty<T>
  void foo2() {}; // expected-note 2 {{candidate function}}
  template <typename T>
  void foo3() requires IsEmpty<T> {}; // expected-note 2 {{candidate function}}
};
struct bar1 : public base {
  using base::foo1;
  using base::foo2;
  using base::foo3;
  template <typename T> requires IsEmpty<T>
  void foo1() {}; // expected-note {{candidate function}}
  template <IsEmpty T>
  void foo2() {}; // expected-note {{candidate function}}
  template <IsEmpty T>
  void foo3() {}; // expected-note {{candidate function}}
};
struct bar2 : public base {
  using base::foo1;
  using base::foo2;
  using base::foo3;
  template <typename T>
  void foo1() requires IsEmpty<T> {}; // expected-note {{candidate function}}
  template <typename T>
  void foo2() requires IsEmpty<T> {}; // expected-note {{candidate function}}
  template <typename T> requires IsEmpty<T>
  void foo3() {}; // expected-note {{candidate function}}
};
void func() {
  bar1{}.foo1<Empty>(); // expected-error {{call to member function 'foo1' is ambiguous}}
  bar1{}.foo2<Empty>(); // expected-error {{call to member function 'foo2' is ambiguous}}
  bar1{}.foo3<Empty>(); // expected-error {{call to member function 'foo3' is ambiguous}}
  bar2{}.foo1<Empty>(); // expected-error {{call to member function 'foo1' is ambiguous}}
  bar2{}.foo2<Empty>(); // expected-error {{call to member function 'foo2' is ambiguous}}
  bar2{}.foo3<Empty>(); // expected-error {{call to member function 'foo3' is ambiguous}}
}
} // namespace same_constraint_at_different_place

namespace more_constrained {
struct base1 { 
  template <class T> Opaque<0> foo() { return Opaque<0>(); }
};
struct derived1 : base1 { 
  using base1::foo;
  template <IsEmpty T> Opaque<1> foo() { return Opaque<1>(); }
};
struct base2 { 
  template <IsEmpty T> Opaque<0> foo() { return Opaque<0>(); }
};
struct derived2 : base2 { 
  using base2::foo;
  template <class T> Opaque<1> foo() { return Opaque<1>(); }
};
void func() {
  expect<0>(derived1{}.foo<int>());
  expect<1>(derived1{}.foo<Empty>());
  expect<0>(derived2{}.foo<Empty>());
  expect<1>(derived2{}.foo<int>());
}
} // namespace more_constrained
} // namespace constrained_members

namespace heads_without_concepts {
struct base {
  template <int N, int M>
  int foo() { return 1; };
};

struct bar : public base {
  using base::foo;
  template <int N> 
  int foo() { return 2; }; // expected-note {{candidate template ignored: substitution failure: too many template arguments for function template 'foo'}}
};

void func() {
  bar f;
  f.foo<10>();
  // FIXME(GH58571): bar::foo should not hide base::foo.
  f.foo<10, 10>(); // expected-error {{no matching member function for call to 'foo'}}
}
} // namespace heads_without_concepts.

namespace GH146614 {

template <typename T>
struct base {
    template <typename A>
    void foo(A x)
        requires (requires{x;})
    {}
};


struct child : base<int> {
  using base<int>::foo;
  template <typename A>
  void foo(A x)
      requires (false)
  {}
};

}

namespace GH198663 {

template <class T>
concept HasIsTransparent = requires { typename T::is_transparent; };

template <class K, class V, class Compare>
struct FlatMapBase {
    using key_compare = Compare;
};

template <class K, class V, class Compare>
struct FlatMap : FlatMapBase<K, V, Compare> {
    using Base = FlatMapBase<K, V, Compare>;

    using typename Base::key_compare;

    void at(const K&) {}
    void at(const K&) const {}
    template <class Other>
    void at(const Other&)
        requires HasIsTransparent<key_compare>
    {}
    template <class Other>
    void at(const Other&) const
        requires HasIsTransparent<key_compare>
    {}
};

template <class T>
struct Transparent {
    T t;
};

struct TransparentComparator {
    using is_transparent = void;

    template <class T>
    bool operator()(const T&, const Transparent<T>&) const;

    template <class T>
    bool operator()(const Transparent<T>&, const T& t) const;

    template <class T>
    bool operator()(const T&, const T&) const;
};

struct NonTransparentComparator {
    template <class T>
    bool operator()(const T&, const Transparent<T>&) const;

    template <class T>
    bool operator()(const Transparent<T>&, const T&) const;

    template <class T>
    bool operator()(const T&, const T&) const;
};

template <class M>
concept CanAt = requires(M m, Transparent<int> k) { m.at(k); };

using TransparentMap = FlatMap<int, double, TransparentComparator>;
using NonTransparentMap = FlatMap<int, double, NonTransparentComparator>;

static_assert(CanAt<TransparentMap>);

static_assert(!CanAt<NonTransparentMap>);

}

namespace GH198663_2 {

template<typename T>
auto mv(T& t) -> T&&;

template<typename S, typename T>
concept does_foo = requires(S s) {
	s.template foo<T>();
};

template<typename S>
struct type {
	S member;
	template<typename T>
	auto foo() -> T requires does_foo<decltype(mv(member)), T>;
};

struct returns_int {
	template<typename T>
	auto foo() -> T;
};

struct nothing {};

static_assert(does_foo<type<returns_int>&, int>);
static_assert(not does_foo<type<nothing>&, int>);

}
