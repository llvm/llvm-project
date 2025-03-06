// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

unsigned long long g;
void fun() { g = 0; }

// CHECK: store i64 0, ptr @g{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: ret void, !dbg [[G2R1:!.*]]

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
