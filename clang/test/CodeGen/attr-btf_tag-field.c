// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64 -g -S -emit-llvm -o - %s | FileCheck %s

#define __tag1 __attribute__((btf_decl_tag("tag1")))
#define __tag2 __attribute__((btf_decl_tag("tag2")))

struct t1 {
  int a __tag1 __tag2;
};

int foo(struct t1 *arg) {
  return arg->a;
}

struct t2 {
  int b:1 __tag1 __tag2;
};

int foo2(struct t2 *arg) {
  return arg->b;
}
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "a", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[#]], size: 32, annotations: ![[ANNOT:[0-9]+]])
// CHECK: ![[ANNOT]] = !{![[TAG1:[0-9]+]], ![[TAG2:[0-9]+]]}
// CHECK: ![[TAG1]] = !{!"btf_decl_tag", !"tag1"}
// CHECK: ![[TAG2]] = !{!"btf_decl_tag", !"tag2"}

// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "b", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[#]], size: 1, flags: DIFlagBitField, extraData: i64 0, annotations: ![[ANNOT]])
