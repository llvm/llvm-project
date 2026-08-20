// RUN: %check_clang_tidy -std=c++20-or-later -expect-clang-tidy-error %s bugprone-std-namespace-modification %t

template <class A, class B> struct O : A, B {};
template <class T> void f() {
  auto a = [] {};
  auto b = [] {};
  O(a, b)();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: error: member 'operator()' found in multiple base classes of different types
}
template void f<int>();
