// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64 -g -S -emit-llvm -o - %s | FileCheck %s

#define __tag1 __attribute__((btf_decl_tag("tag1")))
#define __tag2 __attribute__((btf_decl_tag("tag2")))

struct t1 {
  int a;
};

int __tag1 __tag2 foo(struct t1 *arg) {
  return arg->a;
}

// CHECK: distinct !DISubprogram(name: "foo", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], scopeLine: [[#]], flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: ![[#]], retainedNodes: ![[#]], annotations: ![[ANNOT:[0-9]+]])
// CHECK: ![[ANNOT]] = !{![[TAG1:[0-9]+]], ![[TAG2:[0-9]+]]}
// CHECK: ![[TAG1]] = !{!"btf_decl_tag", !"tag1"}
// CHECK: ![[TAG2]] = !{!"btf_decl_tag", !"tag2"}

int __tag1 __tag2 foo2(struct t1 *arg);
int foo2(struct t1 *arg) {
  return arg->a;
}

// CHECK: distinct !DISubprogram(name: "foo2", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], scopeLine: [[#]], flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: ![[#]], retainedNodes: ![[#]], annotations: ![[ANNOT]])

int __tag1 foo3(struct t1 *arg);
int __tag2 foo3(struct t1 *arg) {
  return arg->a;
}

// CHECK: distinct !DISubprogram(name: "foo3", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], scopeLine: [[#]], flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: ![[#]], retainedNodes: ![[#]], annotations: ![[ANNOT]])

int __tag1 foo4(struct t1 *arg);
int __tag2 foo4(struct t1 *arg);
int foo4(struct t1 *arg) {
  return arg->a;
}

// CHECK: distinct !DISubprogram(name: "foo4", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], scopeLine: [[#]], flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: ![[#]], retainedNodes: ![[#]], annotations: ![[ANNOT]])
