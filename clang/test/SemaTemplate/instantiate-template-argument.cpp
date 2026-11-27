// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify=expected,cxx20
// RUN: %clang_cc1 -std=c++2c -x c++ %s -verify


template<auto T, decltype(T) U>
concept C1 = sizeof(U) >= 4;
// sizeof(U) >= 4 [U = U (decltype(T))]

template<typename Y, char V>
concept C2 = C1<Y{}, V>;
// sizeof(U) >= 4 [U = V (decltype(Y{}))]

template<char W>
constexpr int foo() requires C2<int, W> { return 1; } // #cand1
// sizeof(U) >= 4 [U = W (decltype(int{}))]

template<char X>
constexpr int foo() requires C1<1, X> && true { return 2; } // #cand2
// sizeof(U) >= 4 [U = X (decltype(1))]

static_assert(foo<'a'>() == 2);


template<char Z>
constexpr int foo() requires C2<long long, Z> && true { return 3; } // #cand3
// sizeof(U) >= 4 [U = Z (decltype(long long{}))]

static_assert(foo<'a'>() == 3);
// expected-error@-1{{call to 'foo' is ambiguous}}
// expected-note@#cand2 {{candidate function}}
// expected-note@#cand3 {{candidate function}}


namespace case1 {

template<auto T, decltype(T) U>
concept C1 = sizeof(T) >= 4; // #case1_C1

template<typename Y, char V>
concept C2 = C1<Y{}, V>; // #case1_C2

template<class T, char W>
constexpr int foo() requires C2<T, W> { return 1; } // #case1_foo1

template<class T, char X>
constexpr int foo() requires C1<T{}, X> && true { return 2; } // #case1_foo2

static_assert(foo<char, 'a'>() == 2);
// expected-error@-1{{no matching function for call to 'foo'}}
// expected-note@#case1_foo1{{candidate template ignored: constraints not satisfied [with T = char, W = 'a']}}
// expected-note@#case1_foo1{{because 'C2<char, 'a'>' evaluated to false}}
// expected-note@#case1_C2{{because 'C1<char{}, 'a'>' evaluated to false}}
// expected-note@#case1_C1{{because 'sizeof ('\x00') >= 4' (1 >= 4) evaluated to false}}
// expected-note@#case1_foo2{{candidate template ignored: constraints not satisfied [with T = char, X = 'a']}}
// expected-note@#case1_foo2{{because 'C1<char{}, 'a'>' evaluated to false}}
// expected-note@#case1_C1{{because 'sizeof ('\x00') >= 4' (1 >= 4) evaluated to false}}

static_assert(foo<int, 'a'>() == 2);

}

namespace packs {

template<auto T, decltype(T) U>
concept C1 = sizeof(U) >= 4;

template<typename Y, char V>
concept C2 = C1<Y{}, V>;

template<char... W>
constexpr int foo() requires (C2<int, W> && ...) { return 1; } // #packs-cand1

template<char... X>
constexpr int foo() requires (C1<1, X> && ...) && true { return 2; } // #packs-cand2

static_assert(foo<'a'>() == 2);
// cxx20-error@-1{{call to 'foo' is ambiguous}}
// cxx20-note@#packs-cand1 {{candidate function}}
// cxx20-note@#packs-cand2 {{candidate function}}

}

namespace case2 {
template<auto T> concept C1 = sizeof(decltype(T)) >= 0;
template<typename Y> concept C2 = C1<Y{}>;

template<char W>
constexpr int foo() requires C2<int> { return 1; }

template<char X>
constexpr int foo() requires C1<0> && true { return 2; }

static_assert(foo<0>() == 2);
}

namespace case3 {
template<auto T> concept C1 = sizeof(decltype(T)) >= 0;

template<typename Y> concept C2 = C1<Y{}>;

template<char W>
constexpr int foo() requires C2<int> { return 1; } // #case3_foo1

template<char X>
constexpr int foo() requires C1<1> && true { return 2; } // #case3_foo2

static_assert(foo<0>() == 2);
// expected-error@-1{{call to 'foo' is ambiguous}}
// expected-note@#case3_foo1 {{candidate function}}
// expected-note@#case3_foo2 {{candidate function}}
}
