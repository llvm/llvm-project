
// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

struct Struct { int a, b, c; };
void fun(Struct &a, Struct &b) {
// CHECK: call void @llvm.memcpy{{.*}}, !dbg  [[G1R1:!.*]]
  a = b;
// CHECK: ret void, !dbg [[G2R1:!.*]]
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
