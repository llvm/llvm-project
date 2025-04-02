// RUN: %clang -gkey-instructions %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

unsigned long long g;
void fun() {
// CHECK: store i64 0, ptr @g{{.*}}, !dbg [[G1R1:!.*]]
    g = 0;

// Treat the two assignments as two atoms.
//
// FIXME: Because of the atomGroup implementation the load can only be
// associated with one of the two stores, despite being a good backup
// loction for both.
// CHECK-NEXT: %0 = load i64, ptr @g{{.*}}, !dbg [[G2R2:!.*]]
// CHECK-NEXT: store i64 %0, ptr @g{{.*}}, !dbg [[G3R1:!.*]]
// CHECK-NEXT: store i64 %0, ptr @g{{.*}}, !dbg [[G2R1:!.*]]
    g = g = g;
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
