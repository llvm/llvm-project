// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 \
// RUN:   -emit-llvm -o - %s | FileCheck %s

export module mod;

template <class F> struct Iterator {
  F fn;

  decltype(auto) operator*() {
    return fn(char{});
  }
};

template <typename> void bad_function() {
  auto lam1 = [](char n) { return 0; };
  static auto lam2 = [](char n) { return n; };

  *Iterator{lam1};
  *Iterator{lam2};
}

template <typename> int both_static() {
  static auto lam1 = [](char) { return 1; };
  static auto lam2 = [](char) { return 2; };

  return *Iterator{lam1} + *Iterator{lam2};
}

template <typename> int both_generic() {
  static constexpr auto lam1 = [](auto n) { return 0; };
  static constexpr auto lam2 = [](auto n) { return n; };

  return *Iterator{lam1} + *Iterator{lam2};
}

int repro() {
  bad_function<int>();
  return both_static<int>() + both_generic<int>();
}

// CHECK-DAG: define{{.*}} i32 @_ZNW3mod8IteratorIZS_12bad_functionIiEvvEUlcE_EdeEv(
// CHECK-DAG: define{{.*}} i8 @_ZNW3mod8IteratorIZS_12bad_functionIiEvvEN4lam2MUlcE_EEdeEv(
// CHECK-DAG: define{{.*}} i32 @_ZNW3mod8IteratorIZS_11both_staticIiEivEN4lam1MUlcE_EEdeEv(
// CHECK-DAG: define{{.*}} i32 @_ZNW3mod8IteratorIZS_11both_staticIiEivEN4lam2MUlcE_EEdeEv(
// CHECK-DAG: define{{.*}} i32 @_ZNW3mod8IteratorIZS_12both_genericIiEivEN4lam1MUlT_E_EEdeEv(
// CHECK-DAG: define{{.*}} i8 @_ZNW3mod8IteratorIZS_12both_genericIiEivEN4lam2MUlT_E_EEdeEv(