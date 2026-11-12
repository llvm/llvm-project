// REQUIRES: bpf-registered-target
// RUN: %clang -target bpf -S -emit-llvm -g -O2 -o - %s  | FileCheck %s

#define __decl_tag(x) __attribute__((btf_decl_tag(x)))

extern void foo(int aa __decl_tag("aa_tag"), long bb __decl_tag("bb_tag"));
extern void bar(char cc __decl_tag("cc_tag"));
extern void buz(int __decl_tag("buz_arg_tag"));

void root(void) {
  foo(0, 1);
  bar(0);
  buz(0);
}
// CHECK: [[foo:![0-9]+]] = !DISubprogram(name: "foo", {{.*}}, retainedNodes: [[foo_nodes:![0-9]+]])
// CHECK: [[foo_nodes]] = !{[[aa:![0-9]+]], [[bb:![0-9]+]]}

// CHECK: [[aa]] = !DILocalVariable(name: "aa", arg: 1, scope: [[foo]], {{.*}}, annotations: [[aa_annot:![0-9]+]])
// CHECK: [[aa_annot]] = !{[[aa_tag:![0-9]+]]}
// CHECK: [[aa_tag]] = !{!"btf_decl_tag", !"aa_tag"}

// CHECK: [[bb]] = !DILocalVariable(name: "bb", arg: 2, scope: [[foo]], {{.*}}, annotations: [[bb_annot:![0-9]+]])
// CHECK: [[bb_annot]] = !{[[bb_tag:![0-9]+]]}
// CHECK: [[bb_tag]] = !{!"btf_decl_tag", !"bb_tag"}

// CHECK: [[bar:![0-9]+]] = !DISubprogram(name: "bar", {{.*}}, retainedNodes: [[bar_nodes:![0-9]+]])
// CHECK: [[bar_nodes]] = !{[[cc:![0-9]+]]}

// CHECK: [[cc]] = !DILocalVariable(name: "cc", arg: 1, scope: [[bar]], {{.*}}, annotations: [[cc_annot:![0-9]+]])
// CHECK: [[cc_annot]] = !{[[cc_tag:![0-9]+]]}
// CHECK: [[cc_tag]] = !{!"btf_decl_tag", !"cc_tag"}

// CHECK: [[buz:![0-9]+]] = !DISubprogram(name: "buz", {{.*}}, retainedNodes: [[buz_nodes:![0-9]+]])
// CHECK: [[buz_nodes]] = !{[[buz_arg:![0-9]+]]}

// CHECK: [[buz_arg]] = !DILocalVariable(arg: 1, scope: [[buz]], {{.*}}, annotations: [[buz_arg_annot:![0-9]+]])
// CHECK: [[buz_arg_annot]] = !{[[buz_arg_tag:![0-9]+]]}
// CHECK: [[buz_arg_tag]] = !{!"btf_decl_tag", !"buz_arg_tag"}
