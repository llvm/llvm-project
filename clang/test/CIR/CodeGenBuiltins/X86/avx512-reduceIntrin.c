// RUN: %clang_cc1 -x c -ffreestanding %s -O2 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -fclangir -emit-cir -o - -Wall -Werror | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -x c -ffreestanding %s -O2 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -fclangir -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -x c -ffreestanding %s -O2 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

double test_mm512_reduce_add_pd(__m512d __W, double ExtraAddOp){

  // CIR-LABEL: _mm512_reduce_add_pd
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fadd" %[[R:.*]], %[[V:.*]] : (!cir.double, !cir.vector<8 x !cir.double>) -> !cir.double

  // CIR-LABEL: test_mm512_reduce_add_pd
  // CIR: cir.call @_mm512_reduce_add_pd(%[[VEC:.*]]) : (!cir.vector<8 x !cir.double>) -> !cir.double

  // LLVM-LABEL: test_mm512_reduce_add_pd
  // LLVM: call double @llvm.vector.reduce.fadd.v8f64(double -0.000000e+00, <8 x double> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_add_pd
  // OGCG-NOT: reassoc
  // OGCG: call reassoc {{.*}}double @llvm.vector.reduce.fadd.v8f64(double -0.000000e+00, <8 x double> %{{.*}})
  // OGCG-NOT: reassoc
  return _mm512_reduce_add_pd(__W) + ExtraAddOp;
}

double test_mm512_reduce_mul_pd(__m512d __W, double ExtraMulOp){
  // CIR-LABEL: _mm512_reduce_mul_pd
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmul" %[[R:.*]], %[[V:.*]] : (!cir.double, !cir.vector<8 x !cir.double>) -> !cir.double

  // CIR-LABEL: test_mm512_reduce_mul_pd
  // CIR: cir.call @_mm512_reduce_mul_pd(%[[VEC:.*]]) : (!cir.vector<8 x !cir.double>) -> !cir.double

  // LLVM-LABEL: test_mm512_reduce_mul_pd
  // LLVM: call double @llvm.vector.reduce.fmul.v8f64(double 1.000000e+00, <8 x double> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_mul_pd
  // OGCG-NOT: reassoc
  // OGCG:    call reassoc {{.*}}double @llvm.vector.reduce.fmul.v8f64(double 1.000000e+00, <8 x double> %{{.*}})
  // OGCG-NOT: reassoc
  return _mm512_reduce_mul_pd(__W) * ExtraMulOp;
}


float test_mm512_reduce_add_ps(__m512 __W){
  // CIR-LABEL: _mm512_reduce_add_ps
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fadd" %[[R:.*]], %[[V:.*]] : (!cir.float, !cir.vector<16 x !cir.float>) -> !cir.float

  // CIR-LABEL: test_mm512_reduce_add_ps
  // CIR: cir.call @_mm512_reduce_add_ps(%[[VEC:.*]]) : (!cir.vector<16 x !cir.float>) -> !cir.float

  // LLVM-LABEL: test_mm512_reduce_add_ps
  // LLVM: call float @llvm.vector.reduce.fadd.v16f32(float -0.000000e+00, <16 x float> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_add_ps
  // OGCG: call reassoc {{.*}}float @llvm.vector.reduce.fadd.v16f32(float -0.000000e+00, <16 x float> %{{.*}})
  return _mm512_reduce_add_ps(__W);
}

float test_mm512_reduce_mul_ps(__m512 __W){
  // CIR-LABEL: _mm512_reduce_mul_ps
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmul" %[[R:.*]], %[[V:.*]] : (!cir.float, !cir.vector<16 x !cir.float>) -> !cir.float

  // CIR-LABEL: test_mm512_reduce_mul_ps
  // CIR: cir.call @_mm512_reduce_mul_ps(%[[VEC:.*]]) : (!cir.vector<16 x !cir.float>) -> !cir.float

  // LLVM-LABEL: test_mm512_reduce_mul_ps
  // LLVM: call float @llvm.vector.reduce.fmul.v16f32(float 1.000000e+00, <16 x float> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_mul_ps
  // OGCG:    call reassoc {{.*}}float @llvm.vector.reduce.fmul.v16f32(float 1.000000e+00, <16 x float> %{{.*}})
  return _mm512_reduce_mul_ps(__W);
}
