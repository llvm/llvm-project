// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

template <typename T>
[[clang::noinline]] void foo(T) {}

// Explicit specialization should not inherit noinline
template <>
[[clang::always_inline]] void foo<int>(int) {}

void caller() {
  foo<float>(4.2f); // expect noinline on function
  foo<int>(42); // expect alwaysinline on function
}
// CHECK: define {{.*}}void @_Z3fooIiEvT_({{.*}}) #[[ALWAYSINLINE:[0-9]+]]
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

// Inner function should clobber a conflicting attribute
void inner_fn3();

void outer_fn3() {
  [[clang::noinline]]
  {
    [[clang::always_inline]]
    inner_fn3();
  }
}
// CHECK: call void @_Z9inner_fn3v() #[[ALWAYSINLINE_ONLY:[0-9]+]]

// CHECK: attributes #[[ALWAYSINLINE]] = {
// CHECK-SAME: alwaysinline
// CHECK-NOT:  noinline
// CHECK: attributes #[[NOINLINE]] = {
// CHECK-SAME: noinline
// CHECK-NOT:  alwaysinline

// CHECK: attributes #[[NOINLINE_NOMERGE]] = { noinline nomerge }
// CHECK: attributes #[[ALWAYSINLINE_ONLY]] = { alwaysinline }
