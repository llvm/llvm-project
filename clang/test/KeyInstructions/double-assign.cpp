// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// FIXME: Question - should these all be in the same atomGroup or not?
// FIXME: Because of the atomGroup implementation the load can only be
// associated with one of the two stores, despite being a good backup
// loction for both.

int g;
void a() {
// CHECK: %0 = load i32, ptr @g{{.*}}, !dbg [[G1R2:!.*]]
// CHECK: store i32 %0, ptr %b{{.*}}, !dbg [[G2R1:!.*]]
// CHECK: store i32 %0, ptr %a{{.*}}, !dbg [[G1R1:!.*]]
  int a, b;
  a = b = g;
// CHECK: ret{{.*}}, !dbg [[G3R1:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
