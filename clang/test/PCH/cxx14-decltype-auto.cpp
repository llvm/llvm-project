// Test with pch.
// RUN: %clang_cc1 -emit-pch -std=c++14 -o %t %s
// RUN: %clang_cc1 -include-pch %t -emit-llvm -std=c++14 -o - %s

#ifndef HEADER
#define HEADER

template <typename T>
constexpr decltype(auto) test(T) { return T(); }
class A {};
void k() {  test(A()); }

#else

auto s = test(A());
#endif
