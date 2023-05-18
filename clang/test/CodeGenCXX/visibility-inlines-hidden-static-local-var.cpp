// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fvisibility-inlines-hidden -fvisibility-inlines-hidden-static-local-var %s -emit-llvm -o - | FileCheck %s

#define used __attribute__((used))

used inline void f1() {
  // CHECK: @_ZZ2f1vE6f1_var = linkonce_odr hidden global i32 0
  static int f1_var = 0;
}

__attribute__((visibility("default")))
used inline void f2() {
  // CHECK: @_ZZ2f2vE6f2_var = linkonce_odr global i32 0
  static int f2_var = 0;
}

struct S {
  used void f3() {
    // CHECK: @_ZZN1S2f3EvE6f3_var = linkonce_odr hidden global i32 0
    static int f3_var = 0;
  }

  void f6();
  void f7();
};

used void f4() {
  // CHECK: @_ZZ2f4vE6f4_var = internal global i32 0
  static int f4_var = 0;
}

__attribute__((visibility("default")))
used void f5() {
  // CHECK: @_ZZ2f5vE6f5_var = internal global i32 0
  static int f5_var = 0;
}

used void S::f6() {
  // CHECK: @_ZZN1S2f6EvE6f6_var = internal global i32 0
  static int f6_var = 0;
}

used inline void S::f7() {
  // CHECK: @_ZZN1S2f7EvE6f7_var = linkonce_odr hidden global i32 0
  static int f7_var = 0;
}


struct __attribute__((visibility("default"))) S2 {
  used void f8() {
    // CHECK: @_ZZN2S22f8EvE6f8_var = linkonce_odr hidden global i32 0
    static int f8_var = 0;
  }
};
