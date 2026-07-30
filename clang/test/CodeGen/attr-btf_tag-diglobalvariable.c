// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64 -g -S -emit-llvm -o - %s | FileCheck %s

#define __tag1 __attribute__((btf_decl_tag("tag1")))
#define __tag2 __attribute__((btf_decl_tag("tag2")))

struct t1 {
  int a;
};
struct t1 g1 __tag1 __tag2;

extern struct t1 g2 __tag1 __tag2;
struct t1 g2;

// CHECK: distinct !DIGlobalVariable(name: "g1", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], isLocal: false, isDefinition: true, annotations: ![[ANNOT:[0-9]+]])
// CHECK: distinct !DIGlobalVariable(name: "g2", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], isLocal: false, isDefinition: true, annotations: ![[ANNOT]])
// CHECK: ![[ANNOT]] = !{![[TAG1:[0-9]+]], ![[TAG2:[0-9]+]]}
// CHECK: ![[TAG1]] = !{!"btf_decl_tag", !"tag1"}
// CHECK: ![[TAG2]] = !{!"btf_decl_tag", !"tag2"}

extern struct t1 g3 __tag1;
struct t1 g3 __tag2;

// CHECK: distinct !DIGlobalVariable(name: "g3", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], isLocal: false, isDefinition: true, annotations: ![[ANNOT]])

extern struct t1 g4;
struct t1 g4 __tag1 __tag2;

// CHECK: distinct !DIGlobalVariable(name: "g4", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], isLocal: false, isDefinition: true, annotations: ![[ANNOT]])
