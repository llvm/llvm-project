// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions %s -debug-info-kind=line-tables-only -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Check both the addr-store and value-store are part of the same atom.

void f(int x) {
// CHECK: %call = call noalias noundef nonnull ptr @_Znwm{{.*}}, !dbg [[G1R2_C12:!.*]]
// CHECK: %0 = load i32, ptr %x.addr{{.*}}, !dbg [[G1R2_C20:!.*]]
// CHECK: store i32 %0, ptr %call{{.*}}, !dbg [[G1R1_C12:!.*]]
// CHECK: store ptr %call, ptr %{{.*}}, !dbg [[G1R1_C8:!.*]]
  int *n = new int(x);
// CHECK: %call1 = call noalias noundef nonnull ptr @_Znwm{{.*}}, !dbg [[G2R2_C7:!.*]]
// CHECK: %1 = load i32, ptr %x.addr{{.*}}, !dbg [[G2R2_C15:!.*]]
// CHECK: store i32 %1, ptr %call{{.*}}, !dbg [[G2R1_C7:!.*]]
// CHECK: store ptr %call1, ptr %{{.*}}, !dbg [[G2R1_C5:!.*]]
  n = new int(x);
// CHECK: %2 = load i32, ptr %x.addr{{.*}}, !dbg [[G3R2:!.*]]
// CHECK: %3 = load ptr, ptr %n
// CHECK: store i32 %2, ptr %3{{.*}}, !dbg [[G3R1:!.*]]
  *n = x;
// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G1R2_C12]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R2_C20]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1_C12]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G1R1_C8]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)

// CHECK: [[G2R2_C7]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R2_C15]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1_C7]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G2R1_C5]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)

// CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)

// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
