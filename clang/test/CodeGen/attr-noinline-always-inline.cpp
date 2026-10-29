// RUN: %clang_cc1 -O1 -disable-llvm-passes -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

template <typename T>
[[clang::noinline]] void foo(T) {}

// Explicit specializations should not inherit noinline
template <>
[[clang::always_inline]] void foo<int>(int) {}

template <>
void foo<long>(long) {}

void caller() {
  foo<float>(4.2f); // expect noinline on function
  foo<int>(42); // expect alwaysinline on function
  foo<long>(42L); // expect no noinline attribute
}
// CHECK: define {{.*}}void @_Z3fooIiEvT_({{.*}}) #[[ALWAYSINLINE:[0-9]+]]
// CHECK: define {{.*}}void @_Z3fooIlEvT_({{.*}}) #[[BARE:[0-9]+]]
// CHECK: define {{.*}}void @_Z3fooIfEvT_({{.*}}) #[[NOINLINE:[0-9]+]]

// Inner function should not clobber non-conflicting attributes
void inner_fn();

void outer_fn() {
  [[clang::noinline]]
  {
    [[clang::nomerge]] // unrelated to inlining
    inner_fn();
  }
}
// CHECK: call void @_Z8inner_fnv() #[[NOINLINE_NOMERGE:[0-9]+]]

void inner_fn2();

void outer_fn2() {
  [[clang::nomerge]]
  {
    [[clang::noinline]] // unrelated to nomerge
    inner_fn2();
  }
}
// CHECK: call void @_Z9inner_fn2v() #[[NOINLINE_NOMERGE]]

[[clang::convergent]]
void inner_fn3();

void outer_fn3() {
  [[clang::always_inline]]
  {
    [[clang::noconvergent]] // unrelated to inlining
    inner_fn3();
  }
}
// CHECK: call void @_Z9inner_fn3v() #[[ALWAYSINLINE_ONLY:[0-9]+]]

[[clang::convergent]]
void inner_fn4();

void outer_fn4() {
  [[clang::noconvergent]]
  {
    [[clang::always_inline]] // unrelated to noconvergent
    inner_fn4();
  }
}
// CHECK: call void @_Z9inner_fn4v() #[[ALWAYSINLINE_ONLY]]

// Inner function should clobber a conflicting attribute
void inner_fn5();

void outer_fn5() {
  [[clang::noinline]]
  {
    [[clang::always_inline]]
    inner_fn5();
  }
}
// CHECK: call void @_Z9inner_fn5v() #[[ALWAYSINLINE_ONLY]]

// CHECK: attributes #[[ALWAYSINLINE]] = {
// CHECK-SAME: alwaysinline
// CHECK-NOT:  noinline
// CHECK: attributes #[[BARE]] = {
// CHECK-NOT: alwaysinline
// CHECK-NOT: noinline
// CHECK: attributes #[[NOINLINE]] = {
// CHECK-SAME: noinline
// CHECK-NOT:  alwaysinline

// CHECK: attributes #[[NOINLINE_NOMERGE]] = { noinline nomerge }
// CHECK: attributes #[[ALWAYSINLINE_ONLY]] = { alwaysinline }
