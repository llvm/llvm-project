// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv66 -target-feature +hvxv66 -target-feature +hvx-length128b -emit-llvm -o - %s | FileCheck %s
// REQUIRES: hexagon-registered-target

typedef long HEXAGON_VecPred128 __attribute__((__vector_size__(128)))
  __attribute__((aligned(128)));
typedef long HEXAGON_Vect1024 __attribute__((__vector_size__(128)))
  __attribute__((aligned(128)));
typedef long HEXAGON_Vect2048 __attribute__((__vector_size__(256)))
  __attribute__((aligned(256)));

// CHECK-LABEL: @test1
// CHECK: call <32 x i32> @llvm.hexagon.V6.vaddcarrysat.128B(<32 x i32> %{{[0-9]+}}, <32 x i32> %{{[0-9]+}}, <1024 x i1> %{{[0-9]+}})
HEXAGON_Vect1024 test1(void *in, void *out) {
  HEXAGON_Vect1024 v1, v2;
  HEXAGON_Vect1024 *p;
  HEXAGON_VecPred128 q1;

  p = (HEXAGON_Vect1024 *)in;
  v1 = *p++;
  v2 = *p++;
  q1 = *p++;

  return __builtin_HEXAGON_V6_vaddcarrysat_128B(v1, v2, q1);
}

// CHECK-LABEL: @test26
// CHECK: call <32 x i32> @llvm.hexagon.V6.vrotr.128B(<32 x i32> %{{[0-9]+}}, <32 x i32> %{{[0-9]+}})
HEXAGON_Vect1024 test26(void *in, void *out) {
  HEXAGON_Vect1024 v1, v2;
  HEXAGON_Vect1024 *p;

  p = (HEXAGON_Vect1024 *)in;
  v1 = *p++;
  v2 = *p++;

  return __builtin_HEXAGON_V6_vrotr_128B(v1, v2);
}

// CHECK-LABEL: @test27
// CHECK: call <32 x i32> @llvm.hexagon.V6.vsatdw.128B(<32 x i32> %{{[0-9]+}}, <32 x i32> %{{[0-9]+}})
HEXAGON_Vect1024 test27(void *in, void *out) {
  HEXAGON_Vect1024 v1, v2;
  HEXAGON_Vect1024 *p;

  p = (HEXAGON_Vect1024 *)in;
  v1 = *p++;
  v2 = *p++;

  return __builtin_HEXAGON_V6_vsatdw_128B(v1, v2);
}

// CHECK-LABEL: @test28
// CHECK: call <64 x i32> @llvm.hexagon.V6.vasr.into.128B(<64 x i32> %{{[0-9]+}}, <32 x i32> %{{[0-9]+}}, <32 x i32> %{{[0-9]+}})
HEXAGON_Vect2048 test28(void *in1, void *in2, void *out) {
  HEXAGON_Vect1024 v1, v2;
  HEXAGON_Vect1024 *p1;
  HEXAGON_Vect2048 *p2;
  HEXAGON_Vect2048 vr;

  p1 = (HEXAGON_Vect1024 *)in1;
  v1 = *p1++;
  v2 = *p1++;
  p2 = (HEXAGON_Vect2048 *)in2;
  vr = *p2;

  return __builtin_HEXAGON_V6_vasr_into_128B(vr, v1, v2);
}
