// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-win32 %s -emit-llvm -o - | FileCheck %s

// CHECK: foo{{[^#]*}}#[[ATTRS:[0-9]+]]
__attribute__((no_caller_saved_registers)) void foo() {}
namespace S {
// CHECK: bar{{[^#]*}}#[[ATTRS]]
__attribute__((no_caller_saved_registers)) void bar(int *a) { foo(); }
}

struct St {
  static void baz(int *a) __attribute__((no_caller_saved_registers)) { S::bar(a); }
};

__attribute((no_caller_saved_registers)) void (*foobar)(void);

// CHECK-LABEL: @main
int main(int argc, char **argv) {
  St::baz(&argc);
  // CHECK: [[FOOBAR:%.+]] = load void ()*, void ()** @{{.*}}foobar{{.*}},
  // CHECK-NEXT: call void [[FOOBAR]]() #[[ATTRS1:.+]]
  foobar();
  return 0;
}

// CHECK: baz{{[^#]*}}#[[ATTRS]]

// CHECK: attributes #[[ATTRS]] = {
// CHECK-SAME: "no_caller_saved_registers"
// CHECK-SAME: }
// CHECK: attributes #[[ATTRS1]] = { "no_caller_saved_registers" }
