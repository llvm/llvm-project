// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// NOTE: (return) (g = 1) are two separate atoms. FIXME: is that best?

int g;

// CHECK: _Z1av()
// CHECK: ret void{{.*}}, !dbg [[A_G1R1:!.*]]
void a() { return; }

// CHECK: _Z1bv()
// CHECK: %add = add{{.*}}, !dbg [[B_G1R2:!.*]]
// CHECK: ret i32 %add{{.*}}, !dbg [[B_G1R1:!.*]]
int  b() { return g + 1; }

// CHECK: _Z1cv()
// CHECK: store{{.*}}, !dbg [[C_G2R1:!.*]]
// CHECK: ret i32 1{{.*}}, !dbg [[C_G1R1:!.*]]
int  c() { return g = 1; }

// CHECK: [[A_G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[B_G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[B_G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[C_G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[C_G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
