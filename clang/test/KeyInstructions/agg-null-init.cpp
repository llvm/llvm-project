// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

struct S { void *data[3]; };

// CHECK: _Z1av
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @llvm.memset{{.*}}, !dbg [[A_G1R2:!.*]]
// CHECK-NEXT: ret void, !dbg [[A_G1R1:!.*]]
S a() { return S(); }

// CHECK: _Z1bv
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @llvm.memset{{.*}}, !dbg [[B_G1R2:!.*]]
// CHECK-NEXT: ret void, !dbg [[B_G1R1:!.*]]
S b() { return S{}; }

// CHECK: [[A_G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[A_G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[B_G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[B_G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
