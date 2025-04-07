// RUN: %clang -gkey-instructions -x c++ %s -gmlt -S -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang -gkey-instructions -x c %s -gmlt -S -emit-llvm -o -  \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

typedef struct { int a, b, c; } Struct;
void fun(Struct a) {
// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[G1R1:!.*]]
  Struct b = a;

// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[G2R1:!.*]]
  b = a;
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
