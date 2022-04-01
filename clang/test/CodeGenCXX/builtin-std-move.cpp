// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o - -std=c++17 %s | FileCheck %s --implicit-check-not=@_ZSt4move

namespace std {
  template<typename T> constexpr T &&move(T &val) { return static_cast<T&&>(val); }
  template<typename T> constexpr T &&move_if_noexcept(T &val);
  template<typename T> constexpr T &&forward(T &val);

  // Not the builtin.
  template<typename T, typename U> T move(U source, U source_end, T dest);
}

class T {};
extern "C" void take(T &&);

T a;

// Check emission of a constant-evaluated call.
// CHECK-DAG: @move_a = constant %[[T:.*]]* @a
T &&move_a = std::move(a);
// CHECK-DAG: @move_if_noexcept_a = constant %[[T]]* @a
T &&move_if_noexcept_a = std::move_if_noexcept(a);
// CHECK-DAG: @forward_a = constant %[[T]]* @a
T &forward_a = std::forward<T&>(a);

// Check emission of a non-constant call.
// CHECK-LABEL: define {{.*}} void @test
extern "C" void test(T &t) {
  // CHECK: store %[[T]]* %{{.*}}, %[[T]]** %[[T_REF:[^,]*]]
  // CHECK: %0 = load %[[T]]*, %[[T]]** %[[T_REF]]
  // CHECK: call void @take(%[[T]]* {{.*}} %0)
  take(std::move(t));
  // CHECK: %1 = load %[[T]]*, %[[T]]** %[[T_REF]]
  // CHECK: call void @take(%[[T]]* {{.*}} %1)
  take(std::move_if_noexcept(t));
  // CHECK: %2 = load %[[T]]*, %[[T]]** %[[T_REF]]
  // CHECK: call void @take(%[[T]]* {{.*}} %2)
  take(std::forward<T&&>(t));

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

// CHECK: define {{.*}} i32* @_ZSt4moveIiEOT_RS0_(i32*
