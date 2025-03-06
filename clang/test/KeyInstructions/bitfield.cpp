// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

struct S { int a:3; };

void foo(int x, S s) {
// CHECK: %bf.set = or i8 %bf.clear, %bf.value, !dbg [[G1R2:!.*]]
// CHECK: store i8 %bf.set, ptr %s, align 4, !dbg [[G1R1:!.*]]
  s.a = x;
// CHECK: ret void, !dbg [[G2R1:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
