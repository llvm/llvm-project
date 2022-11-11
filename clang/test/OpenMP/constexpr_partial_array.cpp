// RUN: %clang_cc1 -verify -triple x86_64-pc-windows-msvc19.22.27905 -emit-llvm -o - -fopenmp %s | FileCheck %s
// expected-no-diagnostics

// CHECK: [[C_VAR_VAL:@.+]] = private unnamed_addr constant <{ i8, [26 x i8] }> <{ i8 1, [26 x i8] zeroinitializer }>,
char a;
bool b() {
  static constexpr bool c[27]{1};
  // CHECK: getelementptr inbounds [27 x i8], ptr [[C_VAR_VAL]],
  return c[a];
}
