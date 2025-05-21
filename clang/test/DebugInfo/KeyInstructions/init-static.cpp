// RUN: %clang_cc1 -gkey-instructions %s -debug-info-kind=line-tables-only -emit-llvm -o -\
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

void g(int *a) {
    // CHECK: %2 = load ptr, ptr %a.addr{{.*}}, !dbg [[G1R2:!.*]]
    // CHECK: store ptr %2, ptr @_ZZ1gPiE1b{{.*}}, !dbg [[G1R1:!.*]]
    static int &b = *a;
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)

