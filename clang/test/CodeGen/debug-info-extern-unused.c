// REQUIRES: bpf-registered-target
// RUN: %clang -target bpf -emit-llvm -S -g %s -o - | FileCheck %s

extern char ch;
int test() {
  return 0;
}

int test2() {
  extern char ch2;
  return 0;
}

extern int (*foo)(int);
int test3() {
  return 0;
}

int test4() {
  extern int (*foo2)(int);
  return 0;
}

// CHECK-NOT: distinct !DIGlobalVariable(name: "ch"
// CHECK-NOT: distinct !DIGlobalVariable(name: "ch2"
// CHECK-NOT: distinct !DIGlobalVariable(name: "foo"
// CHECK-NOT: distinct !DIGlobalVariable(name: "foo2"
