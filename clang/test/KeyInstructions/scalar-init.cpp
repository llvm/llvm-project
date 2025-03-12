// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

void a() {
// CHECK: _Z1av()
// CHECK: store i32 0, ptr %A{{.*}}, !dbg [[G1R1:!.*]]
    int A = 0;
// CHECK: %add = add {{.*}}, !dbg [[G2R2:!.*]]
// CHECK: store i32 %add, ptr %B, align 4, !dbg [[G2R1:!.*]]
    int B = 2 * A + 1;
// CHECK: ret{{.*}}, !dbg [[G3R1:!.*]]
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
