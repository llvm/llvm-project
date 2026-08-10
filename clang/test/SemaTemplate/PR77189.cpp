// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s
// expected-no-diagnostics

struct false_type {
	static constexpr bool value = false;
};

struct true_type {
	static constexpr bool value = true;
};

template <auto& Value, int>
struct test : false_type {};

template <auto& Value>
struct test<Value, 0> : true_type {};

int main() {
    static constexpr int v = 42;
    static_assert(test<v, 0>::value);
}
