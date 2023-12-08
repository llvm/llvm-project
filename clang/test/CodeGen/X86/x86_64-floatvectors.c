// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | \
// RUN:   FileCheck %s

// This test validates that the inreg branch generation for __builtin_va_arg 
// does not exceed the alloca size of the type, which can cause the SROA pass to
// eliminate the assignment.

typedef struct { float x, y, z; } vec3f;

double Vec3FTest(__builtin_va_list ap) {
  vec3f vec = __builtin_va_arg(ap, vec3f);
  return vec.x + vec.y + vec.z;
}
// CHECK: define{{.*}} double @Vec3FTest
// CHECK: vaarg.in_reg:
// CHECK: [[Vec3FLoad1:%.*]] = load <2 x float>, ptr
// CHECK: [[Vec3FGEP1:%.*]] = getelementptr inbounds { <2 x float>, float }, ptr {{%.*}}, i32 0, i32 0
// CHECK: store <2 x float> [[Vec3FLoad1]], ptr [[Vec3FGEP1]]
// CHECK: [[Vec3FLoad2:%.*]] = load float, ptr
// CHECK: [[Vec3FGEP2:%.*]] = getelementptr inbounds { <2 x float>, float }, ptr {{%.*}}, i32 0, i32 1
// CHECK: store float [[Vec3FLoad2]], ptr [[Vec3FGEP2]]
// CHECK: vaarg.in_mem:


typedef struct { float x, y, z, q; } vec4f;

double Vec4FTest(__builtin_va_list ap) {
  vec4f vec = __builtin_va_arg(ap, vec4f);
  return vec.x + vec.y + vec.z + vec.q;
}
// CHECK: define{{.*}} double @Vec4FTest
// CHECK: vaarg.in_reg:
// CHECK: [[Vec4FLoad1:%.*]] = load <2 x float>, ptr
// CHECK: [[Vec4FGEP1:%.*]] = getelementptr inbounds { <2 x float>, <2 x float> }, ptr {{%.*}}, i32 0, i32 0
// CHECK: store <2 x float> [[Vec4FLoad1]], ptr [[Vec4FGEP1]]
// CHECK: [[Vec4FLoad2:%.*]] = load <2 x float>, ptr
// CHECK: [[Vec4FGEP2:%.*]] = getelementptr inbounds { <2 x float>, <2 x float> }, ptr {{%.*}}, i32 0, i32 1
// CHECK: store <2 x float> [[Vec4FLoad2]], ptr [[Vec4FGEP2]]
// CHECK: vaarg.in_mem:

typedef struct { double x, y; } vec2d;

double Vec2DTest(__builtin_va_list ap) {
  vec2d vec = __builtin_va_arg(ap, vec2d);
  return vec.x + vec.y;
}
// CHECK: define{{.*}} double @Vec2DTest
// CHECK: vaarg.in_reg:
// CHECK: [[Vec2DLoad1:%.*]] = load double, ptr
// CHECK: [[Vec2DGEP1:%.*]] = getelementptr inbounds { double, double }, ptr {{%.*}}, i32 0, i32 0
// CHECK: store double [[Vec2DLoad1]], ptr [[Vec2DGEP1]]
// CHECK: [[Vec2DLoad2:%.*]] = load double, ptr
// CHECK: [[Vec2DGEP2:%.*]] = getelementptr inbounds { double, double }, ptr {{%.*}}, i32 0, i32 1
// CHECK: store double [[Vec2DLoad2]], ptr [[Vec2DGEP2]]
// CHECK: vaarg.in_mem:

typedef struct {
  float x, y;
  double z;
} vec2f1d;

double Vec2F1DTest(__builtin_va_list ap) {
  vec2f1d vec = __builtin_va_arg(ap, vec2f1d);
  return vec.x + vec.y + vec.z;
}
// CHECK: define{{.*}} double @Vec2F1DTest
// CHECK: vaarg.in_reg:
// CHECK: [[Vec2F1DLoad1:%.*]] = load <2 x float>, ptr
// CHECK: [[Vec2F1DGEP1:%.*]] = getelementptr inbounds { <2 x float>, double }, ptr {{%.*}}, i32 0, i32 0
// CHECK: store <2 x float> [[Vec2F1DLoad1]], ptr [[Vec2F1DGEP1]]
// CHECK: [[Vec2F1DLoad2:%.*]] = load double, ptr
// CHECK: [[Vec2F1DGEP2:%.*]] = getelementptr inbounds { <2 x float>, double }, ptr {{%.*}}, i32 0, i32 1
// CHECK: store double [[Vec2F1DLoad2]], ptr [[Vec2F1DGEP2]]
// CHECK: vaarg.in_mem:

typedef struct {
  double x;
  float y, z;
} vec1d2f;

double Vec1D2FTest(__builtin_va_list ap) {
  vec1d2f vec = __builtin_va_arg(ap, vec1d2f);
  return vec.x + vec.y + vec.z;
}
// CHECK: define{{.*}} double @Vec1D2FTest
// CHECK: vaarg.in_reg:
// CHECK: [[Vec1D2FLoad1:%.*]] = load double, ptr
// CHECK: [[Vec1D2FGEP1:%.*]] = getelementptr inbounds { double, <2 x float> }, ptr {{%.*}}, i32 0, i32 0
// CHECK: store double [[Vec1D2FLoad1]], ptr [[Vec1D2FGEP1]]
// CHECK: [[Vec1D2FLoad2:%.*]] = load <2 x float>, ptr
// CHECK: [[Vec1D2FGEP2:%.*]] = getelementptr inbounds { double, <2 x float> }, ptr {{%.*}}, i32 0, i32 1
// CHECK: store <2 x float> [[Vec1D2FLoad2]], ptr [[Vec1D2FGEP2]]
// CHECK: vaarg.in_mem:

typedef struct {
  float x;
  double z;
} vec1f1d;

double Vec1F1DTest(__builtin_va_list ap) {
  vec1f1d vec = __builtin_va_arg(ap, vec1f1d);
  return vec.x  + vec.z;
}
// CHECK: define{{.*}} double @Vec1F1DTest
// CHECK: vaarg.in_reg:
// CHECK: [[Vec1F1DLoad1:%.*]] = load float, ptr
// CHECK: [[Vec1F1DGEP1:%.*]] = getelementptr inbounds { float, double }, ptr {{%.*}}, i32 0, i32 0
// CHECK: store float [[Vec1F1DLoad1]], ptr [[Vec1F1DGEP1]]
// CHECK: [[Vec1F1DLoad2:%.*]] = load double, ptr
// CHECK: [[Vec1F1DGEP2:%.*]] = getelementptr inbounds { float, double }, ptr {{%.*}}, i32 0, i32 1
// CHECK: store double [[Vec1F1DLoad2]], ptr [[Vec1F1DGEP2]]
// CHECK: vaarg.in_mem:

typedef struct {
  double x;
  float z;
} vec1d1f;

double Vec1D1FTest(__builtin_va_list ap) {
  vec1d1f vec = __builtin_va_arg(ap, vec1d1f);
  return vec.x  + vec.z;
}
// CHECK: define{{.*}} double @Vec1D1FTest
// CHECK: vaarg.in_reg:
// CHECK: [[Vec1D1FLoad1:%.*]] = load double, ptr
// CHECK: [[Vec1D1FGEP1:%.*]] = getelementptr inbounds { double, float }, ptr {{%.*}}, i32 0, i32 0
// CHECK: store double [[Vec1D1FLoad1]], ptr [[Vec1D1FGEP1]]
// CHECK: [[Vec1D1FLoad2:%.*]] = load float, ptr
// CHECK: [[Vec1D1FGEP2:%.*]] = getelementptr inbounds { double, float }, ptr {{%.*}}, i32 0, i32 1
// CHECK: store float [[Vec1D1FLoad2]], ptr [[Vec1D1FGEP2]]
// CHECK: vaarg.in_mem:
