// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

typedef int vint4 __attribute__((ext_vector_type(4)));
typedef short vshort8 __attribute__((ext_vector_type(8)));
typedef float vfloat4 __attribute__((ext_vector_type(4)));
typedef double vdouble4 __attribute__((ext_vector_type(4)));

void test_builtin_elementwise_acos(float f, double d, vfloat4 vf4,
                                   vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_acos
  // LLVM-LABEL: test_builtin_elementwise_acos
  // OGCG-LABEL: test_builtin_elementwise_acos

  // CIR: %{{.*}} = cir.acos %{{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.acos.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.acos.f32(float %{{.*}})
  f = __builtin_elementwise_acos(f);

  // CIR: %{{.*}} = cir.acos %{{.*}} : !cir.double
  // LLVM: %{{.*}} = call double @llvm.acos.f64(double %{{.*}})
  // OGCG: %{{.*}} = call double @llvm.acos.f64(double %{{.*}})
  d = __builtin_elementwise_acos(d);

  // CIR: %{{.*}} = cir.acos %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: %{{.*}} = call <4 x float> @llvm.acos.v4f32(<4 x float> %{{.*}})
  // OGCG: %{{.*}} = call <4 x float> @llvm.acos.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_acos(vf4);

  // CIR: %{{.*}} = cir.acos %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: %{{.*}} = call <4 x double> @llvm.acos.v4f64(<4 x double> %{{.*}})
  // OGCG: %{{.*}} = call <4 x double> @llvm.acos.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_acos(vd4);
}

void test_builtin_elementwise_asin(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_asin
  // LLVM-LABEL: test_builtin_elementwise_asin
  // OGCG-LABEL: test_builtin_elementwise_asin

  // CIR: %{{.*}} = cir.asin %{{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.asin.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.asin.f32(float %{{.*}})
  f = __builtin_elementwise_asin(f);

  // CIR: %{{.*}} = cir.asin %{{.*}} : !cir.double
  // LLVM: %{{.*}} = call double @llvm.asin.f64(double %{{.*}})
  // OGCG: %{{.*}} = call double @llvm.asin.f64(double %{{.*}})
  d = __builtin_elementwise_asin(d);

  // CIR: %{{.*}} = cir.asin %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: %{{.*}} = call <4 x float> @llvm.asin.v4f32(<4 x float> %{{.*}})
  // OGCG: %{{.*}} = call <4 x float> @llvm.asin.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_asin(vf4);

  // CIR: %{{.*}} = cir.asin %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: %{{.*}} = call <4 x double> @llvm.asin.v4f64(<4 x double> %{{.*}})
  // OGCG: %{{.*}} = call <4 x double> @llvm.asin.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_asin(vd4);
}

void test_builtin_elementwise_atan(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_atan
  // LLVM-LABEL: test_builtin_elementwise_atan
  // OGCG-LABEL: test_builtin_elementwise_atan

  // CIR: %{{.*}} = cir.atan %{{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.atan.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.atan.f32(float %{{.*}})
  f = __builtin_elementwise_atan(f);

  // CIR: %{{.*}} = cir.atan %{{.*}} : !cir.double
  // LLVM: %{{.*}} = call double @llvm.atan.f64(double %{{.*}})
  // OGCG: %{{.*}} = call double @llvm.atan.f64(double %{{.*}})
  d = __builtin_elementwise_atan(d);

  // CIR: %{{.*}} = cir.atan %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: %{{.*}} = call <4 x float> @llvm.atan.v4f32(<4 x float> %{{.*}})
  // OGCG: %{{.*}} = call <4 x float> @llvm.atan.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_atan(vf4);

  // CIR: %{{.*}} = cir.atan %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: %{{.*}} = call <4 x double> @llvm.atan.v4f64(<4 x double> %{{.*}})
  // OGCG: %{{.*}} = call <4 x double> @llvm.atan.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_atan(vd4);
}

void test_builtin_elementwise_cos(float f, double d, vfloat4 vf4,
                                     vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_cos
  // LLVM-LABEL: test_builtin_elementwise_cos
  // OGCG-LABEL: test_builtin_elementwise_cos

  // CIR: {{%.*}} = cir.cos {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.cos.f32(float {{%.*}})
  // OGCG: {{%.*}} = call float @llvm.cos.f32(float {{%.*}})
  f = __builtin_elementwise_cos(f);

  // CIR: {{%.*}} = cir.cos {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.cos.f64(double {{%.*}})
  // OGCG: {{%.*}} = call double @llvm.cos.f64(double {{%.*}})
  d = __builtin_elementwise_cos(d);

  // CIR: {{%.*}} = cir.cos {{%.*}} : !cir.vector<4 x !cir.float>
  // LLVM: {{%.*}} = call <4 x float> @llvm.cos.v4f32(<4 x float> {{%.*}})
  // OGCG: {{%.*}} = call <4 x float> @llvm.cos.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_cos(vf4);

  // CIR: {{%.*}} = cir.cos {{%.*}} : !cir.vector<4 x !cir.double>
  // LLVM: {{%.*}} = call <4 x double> @llvm.cos.v4f64(<4 x double> {{%.*}})
  // OGCG: {{%.*}} = call <4 x double> @llvm.cos.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_cos(vd4);
}

void test_builtin_elementwise_fshl(long long int i1, long long int i2,
                                   long long int i3, unsigned short us1,
                                   unsigned short us2, unsigned short us3,
                                   char c1, char c2, char c3,
                                   unsigned char uc1, unsigned char uc2,
                                   unsigned char uc3, vshort8 vi1,
                                   vshort8 vi2, vshort8 vi3, vint4 vu1,
                                   vint4 vu2, vint4 vu3) {
  // CIR-LABEL: test_builtin_elementwise_fshl
  // LLVM-LABEL: test_builtin_elementwise_fshl
  // OGCG-LABEL: test_builtin_elementwise_fshl

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!s64i, !s64i, !s64i) -> !s64i
  // LLVM: %{{.*}} = call i64 @llvm.fshl.i64(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  // OGCG: %{{.*}} = call i64 @llvm.fshl.i64(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  i1 = __builtin_elementwise_fshl(i1, i2, i3);

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!u16i, !u16i, !u16i) -> !u16i
  // LLVM: %{{.*}} = call i16 @llvm.fshl.i16(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
  // OGCG: %{{.*}} = call i16 @llvm.fshl.i16(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
  us1 = __builtin_elementwise_fshl(us1, us2, us3);

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!s8i, !s8i, !s8i) -> !s8i
  // LLVM: %{{.*}} = call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  // OGCG: %{{.*}} = call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  c1 = __builtin_elementwise_fshl(c1, c2, c3);

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!u8i, !u8i, !u8i) -> !u8i
  // LLVM: %{{.*}} = call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  // OGCG: %{{.*}} = call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  uc1 = __builtin_elementwise_fshl(uc1, uc2, uc3);

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>) -> !cir.vector<8 x !s16i>
  // LLVM: %{{.*}} = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // OGCG: %{{.*}} = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vi1 = __builtin_elementwise_fshl(vi1, vi2, vi3);

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>) -> !cir.vector<4 x !s32i>
  // LLVM: %{{.*}} = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // OGCG: %{{.*}} = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vu1 = __builtin_elementwise_fshl(vu1, vu2, vu3);
}

void test_builtin_elementwise_fshr(long long int i1, long long int i2,
                                   long long int i3, unsigned short us1,
                                   unsigned short us2, unsigned short us3,
                                   char c1, char c2, char c3,
                                   unsigned char uc1, unsigned char uc2,
                                   unsigned char uc3, vshort8 vi1,
                                   vshort8 vi2, vshort8 vi3, vint4 vu1,
                                   vint4 vu2, vint4 vu3) {
  // CIR-LABEL: test_builtin_elementwise_fshr
  // LLVM-LABEL: test_builtin_elementwise_fshr
  // OGCG-LABEL: test_builtin_elementwise_fshr

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!s64i, !s64i, !s64i) -> !s64i
  // LLVM: %{{.*}} = call i64 @llvm.fshr.i64(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  // OGCG: %{{.*}} = call i64 @llvm.fshr.i64(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  i1 = __builtin_elementwise_fshr(i1, i2, i3);

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!u16i, !u16i, !u16i) -> !u16i
  // LLVM: %{{.*}} = call i16 @llvm.fshr.i16(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
  // OGCG: %{{.*}} = call i16 @llvm.fshr.i16(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
  us1 = __builtin_elementwise_fshr(us1, us2, us3);

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!s8i, !s8i, !s8i) -> !s8i
  // LLVM: %{{.*}} = call i8 @llvm.fshr.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  // OGCG: %{{.*}} = call i8 @llvm.fshr.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  c1 = __builtin_elementwise_fshr(c1, c2, c3);

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!u8i, !u8i, !u8i) -> !u8i
  // LLVM: %{{.*}} = call i8 @llvm.fshr.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  // OGCG: %{{.*}} = call i8 @llvm.fshr.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  uc1 = __builtin_elementwise_fshr(uc1, uc2, uc3);

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>) -> !cir.vector<8 x !s16i>
  // LLVM: %{{.*}} = call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // OGCG: %{{.*}} = call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vi1 = __builtin_elementwise_fshr(vi1, vi2, vi3);

  // CIR: %{{.*}} = cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>) -> !cir.vector<4 x !s32i>
  // LLVM: %{{.*}} = call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // OGCG: %{{.*}} = call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vu1 = __builtin_elementwise_fshr(vu1, vu2, vu3);
}
