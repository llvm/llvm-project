// RUN: %clang_cc1 -x c -debug-info-kind=limited -triple bpf-linux-gnu -emit-llvm %s -o - | FileCheck %s

extern char ch;
int test() {
  extern short sh;
  return ch + sh;
}

extern char (*foo)(char);
int test2() {
  return foo(0) + ch;
}

// CHECK: distinct !DIGlobalVariable(name: "ch",{{.*}} type: ![[Tch:[0-9]+]], isLocal: false, isDefinition: false
// CHECK: distinct !DIGlobalVariable(name: "sh",{{.*}} type: ![[Tsh:[0-9]+]], isLocal: false, isDefinition: false
// CHECK: ![[Tsh]] = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)

// CHECK: distinct !DIGlobalVariable(name: "foo",{{.*}} type: ![[Tptr:[0-9]+]], isLocal: false, isDefinition: false
// ![[Tptr]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[Tsub:[0-9]+]], size: 64)
// ![[Tsub]] = !DISubroutineType(types: ![[Tproto:[0-9]+]])
// ![[Tproto]] = !{![[Tch]], ![[Tch]]}
// CHECK: ![[Tch]] = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
