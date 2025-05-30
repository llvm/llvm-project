// RUN: %clang_cc1 -gkey-instructions -x c++ -std=c++17 %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -gkey-instructions -x c %s -debug-info-kind=line-tables-only -emit-llvm -o -  \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Perennial question: should the `dec` be in its own source atom or not
// (currently it is).

// We've made the cmp and br separate source atoms for now, to match existing
// behaviour in this case:
// 1. while (
// 2.   int i = --End
// 3.   ) {
// 4.   useValue(i);
// 5. }
// Without Key Instructions we go: 2, 1[, 4, 2, 1]+
// Without separating cmp and br with Key Instructions we'd get:
// 1[, 4, 1]+. If we made the cmp higher precedence than the
// br and had them in the same group, we could get:
// 2, [4, 2]+ which might be nicer. FIXME: do that later.

void a(int A) {
// CHECK: %dec = add nsw i32 %0, -1, !dbg [[G1R2:!.*]]
// CHECK: store i32 %dec, ptr %A.addr{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: %tobool = icmp ne i32 %dec, 0, !dbg [[G2R1:!.*]]
// CHECK: br i1 %tobool, label %while.body, label %while.end, !dbg [[G3R1:!.*]]
    while (--A) { };
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
