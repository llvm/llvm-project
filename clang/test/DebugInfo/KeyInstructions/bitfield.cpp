// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Check assignments to bitfield members are given source atom groups, as this
// has distinct codegen codepath to other variable/member assignments.

struct S { int a:3; };

void foo(int x, S s) {
// CHECK: %bf.set = or i8 %bf.clear, %bf.value, !dbg [[G1R2:!.*]]
// CHECK: store i8 %bf.set, ptr %s, align 4, !dbg [[G1R1:!.*]]
  s.a = x;

// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
