// REQUIRES: bpf-registered-target
// RUN: %clang -target bpf -emit-llvm -S -g %s -o - | FileCheck %s

extern char ch;
extern char ch;
int test() {
  return ch;
}

// CHECK: distinct !DIGlobalVariable(name: "ch",{{.*}} type: ![[T:[0-9]+]], isLocal: false, isDefinition: false
// CHECK-NOT: distinct !DIGlobalVariable(name: "ch"
