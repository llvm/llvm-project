// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK

#if __cplusplus == 199711L
#define static_assert(expr) __extension__ _Static_assert(expr)
#define noexcept throw()
#endif

namespace cwg6 { // cwg6: 2.7
#if __cplusplus >= 201103L
struct Counter {
  int copies;
  constexpr Counter(int copies) : copies(copies) {}
  constexpr Counter(const Counter& other) : copies(other.copies + 1) {}
};

// Passing an lvalue by value makes a non-elidable copy.
constexpr int PassByValue(Counter c) { return c.copies; }
constexpr int PassByValue2(Counter c) { return PassByValue(c); }
constexpr int PassByValue3(Counter c) { return PassByValue2(c); }
static_assert(PassByValue(Counter(0)) == 0, "expect no copies");
static_assert(PassByValue2(Counter(0)) == 1, "expect 1 copy");
static_assert(PassByValue3(Counter(0)) == 2, "expect 2 copies");
#endif

struct A {
  A() noexcept;
  A(const A&) noexcept;
  ~A() noexcept;
};

inline void f(A a) noexcept {}

// CHECK-LABEL: define {{.*}} @_ZN4cwg64callEv
void call() {
  A a;
  // We copy the parameter here, even though object is not mutated by f and
  // otherwise satisfies the criteria for the proposed CWG6 optimization.
  // CHECK: call {{.*}} @_ZN4cwg61AC1ERKS0_(
  // CHECK: call {{.*}} @_ZN4cwg61fENS_1AE(
  f(a);
  // CHECK: call {{.*}} @_ZN4cwg61AD1Ev(
  // CHECK: call {{.*}} @_ZN4cwg61AD1Ev(
}

} // namespace cwg6
