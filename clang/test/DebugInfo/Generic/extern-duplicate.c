// RUN: %clang_cc1 -x c -debug-info-kind=limited -triple bpf-linux-gnu -emit-llvm %s -o - | FileCheck %s

extern char ch;
extern char ch;
int test(void) {
  return ch;
}

// CHECK: distinct !DIGlobalVariable(name: "ch",{{.*}} type: ![[T:[0-9]+]], isLocal: false, isDefinition: false
// CHECK-NOT: distinct !DIGlobalVariable(name: "ch"
