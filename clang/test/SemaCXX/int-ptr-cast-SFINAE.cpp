// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++17

void foo(int* a, int *b) {
  a -= b; // expected-error {{incompatible integer to pointer conversion assigning to 'int *' from}}
}

template<typename T> T declval();
struct true_type { static const bool value = true; };
struct false_type { static const bool value = false; };
template<bool, typename T, typename U> struct select { using type = T; };
template<typename T, typename U> struct select<false, T, U> { using type = U; };


template<typename T>
typename select<(sizeof(declval<T>() -= declval<T>(), 1) != 1), true_type, false_type>::type test(...);
template<typename T> false_type test(...);

template<typename T>
static const auto has_minus_assign = decltype(test<T>())::value;

static_assert(has_minus_assign<int*>, "failed"); // expected-error {{static assertion failed due to requirement 'has_minus_assign<int *>': failed}}
