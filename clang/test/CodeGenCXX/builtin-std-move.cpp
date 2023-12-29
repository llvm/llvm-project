// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o - -std=c++17 %s | FileCheck %s --implicit-check-not=@_ZSt4move

namespace std {
  template<typename T> constexpr T &&move(T &val) { return static_cast<T&&>(val); }
  template<typename T> constexpr T &&move_if_noexcept(T &val);
  template<typename T> constexpr T &&forward(T &val);
  template<typename U, typename T> constexpr T &&forward_like(T &&val);
  template<typename T> constexpr const T &as_const(T &val);

  // Not the builtin.
  template<typename T, typename U> T move(U source, U source_end, T dest);
}

namespace mystd {
  template<typename T> [[clang::behaves_like_std("move")]] constexpr T &&move(T &val) { return static_cast<T&&>(val); }
  template<typename T> [[clang::behaves_like_std("move_if_noexcept")]] constexpr T &&move_if_noexcept(T &val);
  template<typename T> [[clang::behaves_like_std("forward")]] constexpr T &&forward(T &val);
  template<typename U, typename T> [[clang::behaves_like_std("forward_like")]] constexpr T &&forward_like(T &&val);
  template<typename T> [[clang::behaves_like_std("as_const")]] constexpr const T &as_const(T &val);
}

template<typename T> [[clang::behaves_like_std("move")]] constexpr T &&mymove(T &val) { return static_cast<T&&>(val); }
template<typename T> [[clang::behaves_like_std("move_if_noexcept")]] constexpr T &&mymove_if_noexcept(T &val);
template<typename T> [[clang::behaves_like_std("forward")]] constexpr T &&myforward(T &val);
template<typename U, typename T> [[clang::behaves_like_std("forward_like")]] constexpr T &&myforward_like(T &&val);
template<typename T> [[clang::behaves_like_std("as_const")]] constexpr const T &myas_const(T &val);

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
// CHECK-DAG: @forward_like_a = constant ptr @a
T &forward_like_a = std::forward_like<int&>(a);

// CHECK-DAG: @move_a_2 = constant ptr @a
T &&move_a_2 = mystd::move(a);
// CHECK-DAG: @move_if_noexcept_a_2 = constant ptr @a
T &&move_if_noexcept_a_2 = mystd::move_if_noexcept(a);
// CHECK-DAG: @forward_a_2 = constant ptr @a
T &forward_a_2 = mystd::forward<T&>(a);
// CHECK-DAG: @forward_like_a_2 = constant ptr @a
T &forward_like_a_2 = mystd::forward_like<int&>(a);

// CHECK-DAG: @move_a_3 = constant ptr @a
T &&move_a_3 = mymove(a);
// CHECK-DAG: @move_if_noexcept_a_3 = constant ptr @a
T &&move_if_noexcept_a_3 = mymove_if_noexcept(a);
// CHECK-DAG: @forward_a_3 = constant ptr @a
T &forward_a_3 = myforward<T&>(a);
// CHECK-DAG: @forward_like_a_3 = constant ptr @a
T &forward_like_a_3 = myforward_like<int&>(a);

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
  take_lval(std::forward_like<int&>(t));
  // CHECK: %4 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take_lval(ptr {{.*}} %4)
  take_lval(std::as_const<T&&>(t));

  // CHECK: call {{.*}} @_ZSt4moveI1TS0_ET_T0_S2_S1_
  std::move(t, t, t);
}

// CHECK: declare {{.*}} @_ZSt4moveI1TS0_ET_T0_S2_S1_

// CHECK-LABEL: define {{.*}} void @test2
extern "C" void test2(T &t) {
  // CHECK: store ptr %{{.*}}, ptr %[[T_REF:[^,]*]]
  // CHECK: %0 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take(ptr {{.*}} %0)
  take(mystd::move(t));
  // CHECK: %1 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take(ptr {{.*}} %1)
  take(mystd::move_if_noexcept(t));
  // CHECK: %2 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take(ptr {{.*}} %2)
  take(mystd::forward<T&&>(t));
  // CHECK: %3 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take_lval(ptr {{.*}} %3)
  take_lval(mystd::forward_like<int&>(t));
  // CHECK: %4 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take_lval(ptr {{.*}} %4)
  take_lval(mystd::as_const<T&&>(t));
}

// CHECK-LABEL: define {{.*}} void @test3
extern "C" void test3(T &t) {
  // CHECK: store ptr %{{.*}}, ptr %[[T_REF:[^,]*]]
  // CHECK: %0 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take(ptr {{.*}} %0)
  take(mymove(t));
  // CHECK: %1 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take(ptr {{.*}} %1)
  take(mymove_if_noexcept(t));
  // CHECK: %2 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take(ptr {{.*}} %2)
  take(myforward<T&&>(t));
  // CHECK: %3 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take_lval(ptr {{.*}} %3)
  take_lval(myforward_like<int&>(t));
  // CHECK: %4 = load ptr, ptr %[[T_REF]]
  // CHECK: call void @take_lval(ptr {{.*}} %4)
  take_lval(myas_const<T&&>(t));
}

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
