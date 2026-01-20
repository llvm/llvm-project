// no autogeneration since update_cc_test_checks does not support -g
// RUN: %clang_cc1 -triple x86_64-pc-linux -O1 -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define{{.*}} i128 @add(i128 noundef %a)
// CHECK: #dbg_value(i128 %a, ![[DI:.*]], !DIExpression()
__int128_t add(__int128_t a) {
  return a + a;
}

// CHECK: ![[DI]] = !DILocalVariable(name: "a", arg: 1
