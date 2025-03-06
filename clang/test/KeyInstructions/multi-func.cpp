// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Check atomGroup is reset to start at 1 in each function.

void a() {
// CHECK: _Z1av()
// CHECK: store i32 0, ptr %A{{.*}}, !dbg [[A_G1R1:!.*]]
    int A = 0;
// CHECK: store i32 0, ptr %B{{.*}}, !dbg [[A_G2R1:!.*]]
    int B = 0;
// CHECK: ret{{.*}}, !dbg [[A_G3R1:!.*]]
}

void b() {
// CHECK: _Z1bv()
// CHECK: store i32 0, ptr %A{{.*}}, !dbg [[B_G1R1:!.*]]
    int A = 0;
// CHECK: ret{{.*}}, !dbg [[B_G2R1:!.*]]
}

// CHECK: [[A:!.*]] = distinct !DISubprogram(name: "a",
// CHECK: [[A_G1R1]] = !DILocation({{.*}}, scope: [[A]], atomGroup: 1, atomRank: 1)
// CHECK: [[A_G2R1]] = !DILocation({{.*}}, scope: [[A]], atomGroup: 2, atomRank: 1)
// CHECK: [[A_G3R1]] = !DILocation({{.*}}, scope: [[A]], atomGroup: 3, atomRank: 1)
// CHECK: [[B:!.*]] = distinct !DISubprogram(name: "b",
// CHECK: [[B_G1R1]] = !DILocation({{.*}}, scope: [[B]], atomGroup: 1, atomRank: 1)
// CHECK: [[B_G2R1]] = !DILocation({{.*}}, scope: [[B]], atomGroup: 2, atomRank: 1)
