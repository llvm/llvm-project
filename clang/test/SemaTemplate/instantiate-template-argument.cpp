// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

template<auto T, decltype(T) U>
concept C1 = sizeof(U) >= 4;
// sizeof(U) >= 4 [U = U (decltype(T))]

template<typename Y, char V>
concept C2 = C1<Y{}, V>;
// sizeof(U) >= 4 [U = V (decltype(Y{}))]

template<char W>
constexpr int foo() requires C2<int, W> { return 1; }
// sizeof(U) >= 4 [U = W (decltype(int{}))]

template<char X>
// expected-note@+1{{candidate function}}
constexpr int foo() requires C1<1, X> && true { return 2; }
// sizeof(U) >= 4 [U = X (decltype(1))]

static_assert(foo<'a'>() == 2);

template<char Z>
// expected-note@+1{{candidate function}}
constexpr int foo() requires C2<long long, Z> && true { return 3; }
// sizeof(U) >= 4 [U = Z (decltype(long long{}))]

static_assert(foo<'a'>() == 3);
// expected-error@-1{{call to 'foo' is ambiguous}}