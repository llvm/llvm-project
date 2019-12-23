// RUN: %clang_cc1 -x c -debug-info-kind=limited -triple bpf-linux-gnu -emit-llvm %s -o - | FileCheck %s

extern char ch;
int test() {
  return ch;
}

int test2() {
  extern char ch2;
  return ch2;
}

extern int (*foo)(int);
int test3() {
  return foo(0);
}

// CHECK: distinct !DIGlobalVariable(name: "ch",{{.*}} type: ![[CHART:[0-9]+]], isLocal: false, isDefinition: false
// CHECK: distinct !DIGlobalVariable(name: "ch2",{{.*}} type: ![[CHART]], isLocal: false, isDefinition: false
// CHECK: ![[CHART]] = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)

// CHECK: distinct !DIGlobalVariable(name: "foo",{{.*}} type: ![[FUNC:[0-9]+]], isLocal: false, isDefinition: false)
// CHECK: ![[FUNC]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[SUB:[0-9]+]], size: 64)
// CHECK: ![[SUB]] = !DISubroutineType(types: ![[TYPES:[0-9]+]])
// CHECK: ![[TYPES]] = !{![[BASET:[0-9]+]], ![[BASET]]}
// CHECK: ![[BASET]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
