// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64 -g -S -emit-llvm -o - %s | FileCheck %s

#define __tag1 __attribute__((btf_decl_tag("tag1")))
#define __tag2 __attribute__((btf_decl_tag("tag2")))

struct t1 {
  int a;
};

int foo(struct t1 __tag1 __tag2 *arg) {
  return arg->a;
}

// CHECK: !DILocalVariable(name: "arg", arg: 1, scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], annotations: ![[ANNOT:[0-9]+]])
// CHECK: ![[ANNOT]] = !{![[TAG1:[0-9]+]], ![[TAG2:[0-9]+]]}
// CHECK: ![[TAG1]] = !{!"btf_decl_tag", !"tag1"}
// CHECK: ![[TAG2]] = !{!"btf_decl_tag", !"tag2"}
