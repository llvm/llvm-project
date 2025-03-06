// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

void g(int *a) {
// CHECK: %2 = load ptr, ptr %a.addr{{.*}}, !dbg [[G1R2:!.*]]
// CHECK: store ptr %2, ptr @_ZZ1gPiE1b{{.*}}, !dbg [[G1R1:!.*]]
  static int &b = *a;
// CHECK: ret void, !dbg [[G2R2:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
