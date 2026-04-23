// RUN: %clang_cc1 -std=c++20 -fblocks -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

template<void (^B)()> void f() {}

void test_literal() {
  // CHECK: call void @_Z1fIXcvU13block_pointerFvvEadUb_EEvv()
  f<^{}>();
}

// CHECK: define internal void @_Z1fIXcvU13block_pointerFvvEadUb_EEvv()

constexpr void (^global_block)() = ^{};
void test_global() {
  // CHECK: call void @_Z1fIXcvU13block_pointerFvvEadUb0_EEvv()
  f<global_block>();
}

template<int (^B)(int)> struct S {
  static int call(int x) { return B(x); }
};

int test_param(int x) {
  // CHECK: call noundef i32 @_ZN1SIXadUb1_EE4callEi(i32 noundef %0)
  return S<^(int x) { return x + 1; }>::call(x);
}

void test_nullptr() {
  // CHECK: call void @_Z1fILU13block_pointerFvvE0EEvv()
  f<nullptr>();
}

namespace TestNamespace {
  template<void (^B)()> struct S {
    static void call() { B(); }
  };
  void test_namespace() {
    // CHECK: call void @_ZN13TestNamespace1SIXadUb2_EE4callEv()
    S<^{}>::call();
  }
}

template<void (^B)() = ^{}>
void f_default() { B(); }

void test_default() {
  // CHECK: call void @_Z9f_defaultIXcvU13block_pointerFvvEadUb_EEvv()
  f_default();
}

struct Structural {
  void (^b)();
};

template<Structural s>
void f_struct() {
  s.b();
}

void test_struct() {
  // CHECK: call void @_Z8f_structIXtl10StructuraladUb3_EEEvv()
  f_struct<Structural{^{}}>();
}

template<void (^...Blocks)()>
void f_variadic() {
  (Blocks(), ...);
}

void test_variadic() {
  // CHECK: call void @_Z10f_variadicIJXcvU13block_pointerFvvEadUb4_EXcvS1_adUb5_EEEvv()
  f_variadic<^{}, ^{}>();
}

template<auto B>
void f_auto() {
  B();
}

void test_auto() {
  // CHECK: call void @_Z6f_autoITnDaXcvU13block_pointerFvvEadUb6_EEvv()
  f_auto<^{}>();
}
