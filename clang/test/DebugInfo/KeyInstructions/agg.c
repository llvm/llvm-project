// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - -fenable-matrix \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - -fenable-matrix \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

__attribute__((ext_vector_type(1))) char c;
typedef float m5x5 __attribute__((matrix_type(5, 5)));
m5x5 m;
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

// CHECK: %3 = load <25 x float>, ptr @m, align 4
// CHECK: %matins = insertelement <25 x float> %3, float 0.000000e+00, i64 0, !dbg [[G4R2:!.*]]
// CHECK: store <25 x float> %matins, ptr @m{{.*}}, !dbg [[G4R1:!.*]]
  m[0][0] = 0;

// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G4R2]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 2)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
