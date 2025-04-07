// RUN: %clang -gkey-instructions %s -gmlt -gcolumn-info -S -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

struct S { int a:3; };

void foo(int x, S s) {
// CHECK: %bf.set = or i8 %bf.clear, %bf.value, !dbg [[G1R2:!.*]]
// CHECK: store i8 %bf.set, ptr %s, align 4, !dbg [[G1R1:!.*]]
  s.a = x;
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
