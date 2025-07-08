// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

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
// expected-note@+1{{candidate function}}
constexpr int foo() requires C1<1, X> && true { return 2; } // #cand2
// sizeof(U) >= 4 [U = X (decltype(1))]

static_assert(foo<'a'>() == 2);
// expected-error@-1 {{call to 'foo' is ambiguous}}
// expected-note@#cand1 {{candidate function}}
// expected-note@#cand2 {{candidate function}}

template<char Z>
constexpr int foo() requires C2<long long, Z> && true { return 3; } // #cand3
// sizeof(U) >= 4 [U = Z (decltype(long long{}))]

static_assert(foo<'a'>() == 3);
// expected-error@-1{{call to 'foo' is ambiguous}}
// expected-note@#cand1 {{candidate function}}
// expected-note@#cand3 {{candidate function}}
