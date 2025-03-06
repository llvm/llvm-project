// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// FIXME: Perennial quesiton: should the `dec` be in its own source atom or not
// (currently it is).

void a(int A) {
// CHECK: %dec = add nsw i32 %0, -1, !dbg [[G1R2:!.*]]
// CHECK: store i32 %dec, ptr %A.addr{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: %tobool = icmp ne i32 %dec, 0, !dbg [[G2R2:!.*]]
// CHECK: br i1 %tobool, label %while.body, label %while.end, !dbg [[G2R1:!.*]]
    while (--A) { };
// CHECK: ret{{.*}}, !dbg [[G3R1:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
