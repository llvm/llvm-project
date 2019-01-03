// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv66 -target-feature +hvxv66 -target-feature +hvx-length64b -emit-llvm -o - %s | FileCheck %s
// REQUIRES: hexagon-registered-target

// CHECK-LABEL: @test1
// CHECK: call i32 @llvm.hexagon.M2.mnaci(i32 %0, i32 %1, i32 %2)
int test1(int rx, int rs, int rt) {
  return __builtin_HEXAGON_M2_mnaci(rx, rs, rt);
}

// CHECK-LABEL: @test2
// CHECK: call double @llvm.hexagon.F2.dfadd(double %0, double %1)
double test2(double rss, double rtt) {
  return __builtin_HEXAGON_F2_dfadd(rss, rtt);
}

// CHECK-LABEL: @test3
// CHECK: call double @llvm.hexagon.F2.dfsub(double %0, double %1)
double test3(double rss, double rtt) {
  return __builtin_HEXAGON_F2_dfsub(rss, rtt);
}

// CHECK-LABEL: @test4
// CHECK: call i32 @llvm.hexagon.S2.mask(i32 1, i32 2)
int test4() {
  return __builtin_HEXAGON_S2_mask(1, 2);
}

typedef long HEXAGON_VecPred64 __attribute__((__vector_size__(64)))
  __attribute__((aligned(64)));
typedef long HEXAGON_Vect512 __attribute__((__vector_size__(64)))
  __attribute__((aligned(64)));
typedef long HEXAGON_Vect1024 __attribute__((__vector_size__(128)))
  __attribute__((aligned(128)));

// CHECK-LABEL: @test5
// CHECK: call <16 x i32> @llvm.hexagon.V6.vaddcarrysat(<16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}}, <512 x i1> %{{[0-9]+}})
HEXAGON_Vect512 test5(void *in, void *out) {
  HEXAGON_Vect512 v1, v2;
  HEXAGON_Vect512 *p;
  HEXAGON_VecPred64 q1;

  p = (HEXAGON_Vect512 *)in;
  v1 = *p++;
  v2 = *p++;
  q1 = *p++;

  return __builtin_HEXAGON_V6_vaddcarrysat(v1, v2, q1);
}

// CHECK-LABEL: @test6
// CHECK: call <16 x i32> @llvm.hexagon.V6.vrotr(<16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}})
HEXAGON_Vect512 test6(void *in, void *out) {
  HEXAGON_Vect512 v1, v2;
  HEXAGON_Vect512 *p;

  p = (HEXAGON_Vect512 *)in;
  v1 = *p++;
  v2 = *p++;

  return __builtin_HEXAGON_V6_vrotr(v1, v2);
}

// CHECK-LABEL: @test7
// CHECK: call <16 x i32> @llvm.hexagon.V6.vsatdw(<16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}})
HEXAGON_Vect512 test7(void *in, void *out) {
  HEXAGON_Vect512 v1, v2;
  HEXAGON_Vect512 *p;

  p = (HEXAGON_Vect512 *)in;
  v1 = *p++;
  v2 = *p++;

  return __builtin_HEXAGON_V6_vsatdw(v1, v2);
}

// CHECK-LABEL: @test8
// CHECK: call <32 x i32> @llvm.hexagon.V6.vasr.into(<32 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}})
HEXAGON_Vect1024 test8(void *in1, void *in2, void *out) {
  HEXAGON_Vect512 v1, v2;
  HEXAGON_Vect512 *p1;
  HEXAGON_Vect1024 *p2;
  HEXAGON_Vect1024 vr;

  p1 = (HEXAGON_Vect512 *)in1;
  v1 = *p1++;
  v2 = *p1++;
  p2 = (HEXAGON_Vect1024 *)in2;
  vr = *p2;

  return __builtin_HEXAGON_V6_vasr_into(vr, v1, v2);
}
