// RUN: %clang_cc1 -std=c++20 %s -emit-llvm -o -

namespace std {

template <typename T1, typename T2>
struct pair {
	T1 first;
	T2 second;

	// Constructor needed so this reproduces the std::array<std::pair>
	// initialization from the original report.
	constexpr
	pair(const T1& a, const T2& b)
	: first(a), second(b)
	{}
};

template <typename T, unsigned long N>
struct array {
	T elems[N];
};
} // namespace std

// Nested aggregate containing an array of aggregates.
struct Inner {
  int x;
};

struct Outer {
	Inner arr[2];
};

template <typename T>
struct S {
	S() : m({{1}, {2}}) {}
	Outer m;
};

template struct S<int>;

// std::array<std::pair>-style initialization.
template <typename T>
struct S2 {
	S2() : a({{1, 2}}) {}
	std::array<std::pair<int, int>, 1> a;
};

template struct S2<int>;

// Designated initializer.
struct Point {
	int x;
	int y;
};

struct Config {
	Point x;
};

template <typename T>
struct Designated {
	Designated() : cfg({.x = 10, .y = 5}) {}
	Config cfg;
};

template struct Designated<int>;

// String literal initializer.
// Not affected by this fix, but kept as a regression
// for another aggregate initialization path.
struct Buffer {
	char data[10];
};

template <typename T>
struct String {
	String() : buf("hello") {}
	Buffer buf;
};

template struct String<int>;

// Parenthesized aggregate with multiple arguments.
struct First {
	int x;
};

struct Second {
	First a;
	int y;
};

template <typename T>
struct Nested {
	Nested() : sec({1}, 2) {}
	Second sec;
};

template struct Nested<int>;
