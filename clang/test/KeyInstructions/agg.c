// RUN: %clang -gkey-instructions -x c++ %s -gmlt -S -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang -gkey-instructions -x c %s -gmlt -S -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

__attribute__((ext_vector_type(1))) char c;
typedef struct { int a, b, c; } Struct;
void fun(Struct a) {
// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[G1R1:!.*]]
  Struct b = a;

// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[G2R1:!.*]]
  b = a;

// CHECK: %2 = load <1 x i8>, ptr @c
// CHECK: %vecins = insertelement <1 x i8> %2, i8 0, i32 0, !dbg [[G3R2:!.*]]
// CHECK: store <1 x i8> %vecins, ptr @c{{.*}}, !dbg [[G3R1:!.*]]
  c[0] = 0;
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
