// RUN: %clang -gkey-instructions %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

unsigned long long g;
void fun() { g = 0; }

// CHECK: store i64 0, ptr @g{{.*}}, !dbg [[G1R1:!.*]]

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
