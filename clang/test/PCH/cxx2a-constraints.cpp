// RUN: %clang_cc1 -std=c++2a -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++2a -include-pch %t -verify %s

// RUN: %clang_cc1 -std=c++2a -emit-pch -fpch-instantiate-templates %s -o %t
// RUN: %clang_cc1 -std=c++2a -include-pch %t -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template<typename T, typename U = char>
concept SizedLike = sizeof(T) == sizeof(U);

template <class T> void f(T) requires (sizeof(int) == sizeof(T)) {}
template <class T> void f(T) requires (sizeof(char) == sizeof(T)) {}

template <class T> requires (sizeof(int) == sizeof(T)) void g(T) {}
template <class T> requires (sizeof(char) == sizeof(T)) void g(T) {}

template <SizedLike<int> T> void h(T) {}
template <SizedLike<char> T> void h(T) {}

template <SizedLike<int> T> void i(T) {}
template <SizedLike T> void i(T) {}

void j(SizedLike<int> auto ...ints) {}

#else /*included pch*/

int main() {
  (void)f('1');
  (void)f(1);
  (void)g('1');
  (void)g(1);
  (void)h('1');
  (void)h(1);
  (void)i('1');
  (void)i(1);
  (void)j(1, 2, 3);
}

#endif // HEADER
