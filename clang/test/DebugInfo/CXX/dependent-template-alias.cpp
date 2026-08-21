// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone -gtemplate-alias %s -gsimple-template-names=simple \
// RUN: | FileCheck %s

template <int>
using A = int;

template<int I>
struct S {
  using AA = A<I>;
  AA aa;
};

S<0> s;

// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "aa", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[AA:[0-9]+]], size: 32)
// CHECK: [[AA]] = !DIDerivedType(tag: DW_TAG_typedef, name: "AA", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[A:[0-9]+]])
// CHECK: [[A]] = !DIDerivedType(tag: DW_TAG_template_alias, name: "A", file: ![[#]], line: [[#]], baseType: ![[int:[0-9]+]], extraData: ![[#]])
// CHECK: [[int]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
