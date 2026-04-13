// RUN: %clang_cc1 -std=c++20 -fblocks -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

template<void (^B)()> void f() {}

// CHECK: define{{.*}} void @_Z12test_literalv()
void test_literal() {
  // CHECK: call void @_Z1fIXcvU13block_pointerFvvEadUb_EEvv()
  f<^{}>();
}

constexpr void (^global_block)() = ^{};
// CHECK: define{{.*}} void @_Z11test_globalv()
void test_global() {
  // CHECK: call void @_Z1fIXcvU13block_pointerFvvEadUb0_EEvv()
  f<global_block>();
}

template<int (^B)(int)> struct S {
  static int call(int x) { return B(x); }
};

// CHECK: define{{.*}} i32 @_Z10test_parami(i32 noundef %x)
int test_param(int x) {
  // CHECK: call noundef i32 @_ZN1SIXadUb1_EE4callEi(i32 noundef %0)
  return S<^(int x) { return x + 1; }>::call(x);
}
