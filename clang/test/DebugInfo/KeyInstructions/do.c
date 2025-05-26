// RUN: %clang_cc1 -gkey-instructions -x c++ -std=c++17 %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -gkey-instructions -x c %s -debug-info-kind=line-tables-only -emit-llvm -o -  \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Perennial question: should the `dec` be in its own source atom or not
// (currently it is).

// Another question - we've made the cmp and br separate source atoms for
// now, to match existing behaviour in this case:
// 1. do {
// 2.   something();
// 3. }
// 4. while (--A);
// Non key instruction behaviour is: 2, 4[, 3, 2, 4]+
// The cond br is associated with the brace on line 3 and the cmp is line 4;
// if they were in the same atom group we'd step just: 2, 3[, 2, 3]+
// FIXME: We could arguably improve the behaviour by making them the same
// group but having the cmp higher precedence, resulting in: 2, 4[, 2, 4]+.

void a(int A) {
// CHECK: %dec = add nsw i32 %0, -1, !dbg [[G1R2:!.*]]
// CHECK: store i32 %dec, ptr %A.addr{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: %tobool = icmp ne i32 %dec, 0, !dbg [[G2R1:!.*]]
// CHECK: br i1 %tobool, label %do.body, label %do.end, !dbg [[G3R1:!.*]], !llvm.loop
    do { } while (--A);
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
