// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64 -g -S -emit-llvm -o - %s | FileCheck %s

#define __tag1 __attribute__((btf_decl_tag("tag1")))
#define __tag2 __attribute__((btf_decl_tag("tag2")))

struct __tag1 __tag2 t1;
struct t1 {
  int a;
};

int foo(struct t1 *arg) {
  return arg->a;
}

// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: ![[#]], line: [[#]], size: 32, elements: ![[#]], annotations: ![[ANNOT:[0-9]+]])
// CHECK: ![[ANNOT]] = !{![[TAG1:[0-9]+]], ![[TAG2:[0-9]+]]}
// CHECK: ![[TAG1]] = !{!"btf_decl_tag", !"tag1"}
// CHECK: ![[TAG2]] = !{!"btf_decl_tag", !"tag2"}

struct __tag1 t2;
struct __tag2 t2 {
  int a;
};

int foo2(struct t2 *arg) {
  return arg->a;
}

// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2", file: ![[#]], line: [[#]], size: 32, elements: ![[#]], annotations: ![[ANNOT]])

struct __tag1 t3;
struct t3 {
  int a;
} __tag2;

int foo3(struct t3 *arg) {
  return arg->a;
}

// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t3", file: ![[#]], line: [[#]], size: 32, elements: ![[#]], annotations: ![[ANNOT]])

struct t4;
struct t4 {
  int a;
} __tag1 __tag2;

int foo4(struct t4 *arg) {
  return arg->a;
}

// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t4", file: ![[#]], line: [[#]], size: 32, elements: ![[#]], annotations: ![[ANNOT]])
