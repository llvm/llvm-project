// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o - -std=c++17 %s | FileCheck %s --implicit-check-not=@_ZSt4move

namespace std {
  template<typename T> constexpr T &&move(T &val) { return static_cast<T&&>(val); }
  template<typename T> constexpr T &&move_if_noexcept(T &val);
  template<typename T> constexpr T &&forward(T &val);
  template<typename T> constexpr const T &as_const(T &val);

  // Not the builtin.
  template<typename T, typename U> T move(U source, U source_end, T dest);
}

class T {};
extern "C" void take(T &&);
extern "C" void take_lval(const T &);

T a;

// Check emission of a constant-evaluated call.
// CHECK-DAG: @move_a = constant ptr @a
T &&move_a = std::move(a);
// CHECK-DAG: @move_if_noexcept_a = constant ptr @a
T &&move_if_noexcept_a = std::move_if_noexcept(a);
// CHECK-DAG: @forward_a = constant ptr @a
T &forward_a = std::forward<T&>(a);

// Check emission of a non-constant call.
// CHECK-LABEL: define {{.*}} void @test
extern "C" void test(T &t) {
  // CHECK: store ptr %{{.*}}, ptr %[[T_REF:[^,]*]]
  // CHECK: %0 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take(ptr {{.*}} %0)
  take(std::move(t));
  // CHECK: %1 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take(ptr {{.*}} %1)
  take(std::move_if_noexcept(t));
  // CHECK: %2 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take(ptr {{.*}} %2)
  take(std::forward<T&&>(t));
  // CHECK: %3 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take_lval(ptr {{.*}} %3)
  take_lval(std::as_const<T&&>(t));

  // CHECK: call {{.*}} @_ZSt4moveI1TS0_ET_T0_S2_S1_
  std::move(t, t, t);
}

// CHECK: declare {{.*}} @_ZSt4moveI1TS0_ET_T0_S2_S1_

// Check that we instantiate and emit if the address is taken.
// CHECK-LABEL: define {{.*}} @use_address
extern "C" void *use_address() {
  // CHECK: ret {{.*}} @_ZSt4moveIiEOT_RS0_
  return (void*)&std::move<int>;
}

// CHECK: define {{.*}} ptr @_ZSt4moveIiEOT_RS0_(ptr

extern "C" void take_const_int_rref(const int &&);
// CHECK-LABEL: define {{.*}} @move_const_int(
extern "C" void move_const_int() {
  // CHECK: store i32 5, ptr %[[N_ADDR:[^,]*]]
  const int n = 5;
  // CHECK: call {{.*}} @take_const_int_rref(ptr {{.*}} %[[N_ADDR]])
  take_const_int_rref(std::move(n));
}
