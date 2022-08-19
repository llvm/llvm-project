// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

template <typename... Args> requires ((sizeof(Args) == 1), ...)
// expected-note@-1 {{because '(sizeof(int) == 1) , (sizeof(char) == 1) , (sizeof(int) == 1)' evaluated to false}}
void f1(Args&&... args) { }
// expected-note@-1 {{candidate template ignored: constraints not satisfied [with Args = <int, char, int>]}}

using f11 = decltype(f1('a'));
using f12 = decltype(f1(1, 'b'));
using f13 = decltype(f1(1, 'b', 2));
// expected-error@-1 {{no matching function for call to 'f1'}}

template <typename... Args>
void f2(Args&&... args) requires ((sizeof(args) == 1), ...) { }
// expected-note@-1 {{candidate template ignored: constraints not satisfied [with Args = <int, char, int>]}}
// expected-note@-2 {{because '(sizeof (args) == 1) , (sizeof (args) == 1) , (sizeof (args) == 1)' evaluated to false}}

using f21 = decltype(f2('a'));
using f22 = decltype(f2(1, 'b'));
using f23 = decltype(f2(1, 'b', 2));
// expected-error@-1 {{no matching function for call to 'f2'}}

template <typename... Args> requires ((sizeof(Args) == 1), ...)
// expected-note@-1 {{because '(sizeof(int) == 1) , (sizeof(char) == 1) , (sizeof(int) == 1)' evaluated to false}}
void f3(Args&&... args) requires ((sizeof(args) == 1), ...) { }
// expected-note@-1 {{candidate template ignored: constraints not satisfied [with Args = <int, char, int>]}}

using f31 = decltype(f3('a'));
using f32 = decltype(f3(1, 'b'));
using f33 = decltype(f3(1, 'b', 2));
// expected-error@-1 {{no matching function for call to 'f3'}}

template<typename T>
struct S {
	template<typename U>
	static constexpr auto f(U const index) requires(index, true) {
		return true;
	}
};

static_assert(S<void>::f(1));

constexpr auto value = 0;

template<typename T>
struct S2 {
  template<typename = void> requires(value, true)
  static constexpr auto f() requires(value, true) {
  }
};

static_assert((S2<int>::f(), true));

template<typename T>
struct S3 {
	template<typename... Args> requires true
	static constexpr void f(Args...) { }
};

static_assert((S3<int>::f(), true));

template<typename T>
struct S4 {
    template<typename>
    constexpr void foo() requires (decltype(this)(), true) { }
    constexpr void goo() requires (decltype(this)(), true) { }
};

static_assert((S4<int>{}.foo<int>(), S4<int>{}.goo(), true));
