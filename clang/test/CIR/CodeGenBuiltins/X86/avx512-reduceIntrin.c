// RUN: %clang_cc1 -x c -ffreestanding %s -O2 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -fclangir -emit-cir -o - -Wall -Werror | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -x c -ffreestanding %s -O2 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -fclangir -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -x c -ffreestanding %s -O2 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -menable-no-infs -fclangir -emit-cir -o - -Wall -Werror | FileCheck %s --check-prefix=CIR-NINF
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -menable-no-infs -fclangir -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=LLVM-NINF
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -menable-no-infs -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=LLVM-NINF

#include <immintrin.h>

double test_mm512_reduce_add_pd(__m512d __W, double ExtraAddOp){

  // CIR-LABEL: test_mm512_reduce_add_pd
  // CIR: cir.call @_mm512_reduce_add_pd(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<8 x !cir.double>{{.*}}) -> !cir.double

  // CIR-LABEL: cir.func{{.*}} @_mm512_reduce_add_pd(
  // CIR: cir.vec.reduce(fadd, %[[V:.*]], %[[R:.*]]) : (!cir.vector<8 x !cir.double>, !cir.double) -> !cir.double {fastmath_flags = #cir.fastmath<reassoc>}
  // CIR-NINF-LABEL: cir.func{{.*}} @_mm512_reduce_add_pd(
  // CIR-NINF: cir.vec.reduce(fadd, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<ninf, reassoc>}

  // LLVM-LABEL: test_mm512_reduce_add_pd
  // LLVM: call reassoc double @llvm.vector.reduce.fadd.v8f64(double -0.000000e+00, <8 x double> %{{.*}})
  // LLVM-NINF-LABEL: define {{.*}} @test_mm512_reduce_add_pd(
  // LLVM-NINF: call reassoc ninf {{.*}}double @llvm.vector.reduce.fadd.v8f64(

  // OGCG-LABEL: test_mm512_reduce_add_pd
  // OGCG-NOT: reassoc
  // OGCG: call reassoc {{.*}}double @llvm.vector.reduce.fadd.v8f64(double -0.000000e+00, <8 x double> %{{.*}})
  // OGCG-NOT: reassoc
  return _mm512_reduce_add_pd(__W) + ExtraAddOp;
}

double test_mm512_reduce_mul_pd(__m512d __W, double ExtraMulOp){
  // CIR-LABEL: test_mm512_reduce_mul_pd
  // CIR: cir.call @_mm512_reduce_mul_pd(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<8 x !cir.double>{{.*}}) -> !cir.double

  // CIR-LABEL: cir.func{{.*}} @_mm512_reduce_mul_pd(
  // CIR: cir.vec.reduce(fmul, %[[V:.*]], %[[R:.*]]) : (!cir.vector<8 x !cir.double>, !cir.double) -> !cir.double {fastmath_flags = #cir.fastmath<reassoc>}
  // CIR-NINF-LABEL: cir.func{{.*}} @_mm512_reduce_mul_pd(
  // CIR-NINF: cir.vec.reduce(fmul, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<ninf, reassoc>}

  // LLVM-LABEL: test_mm512_reduce_mul_pd
  // LLVM: call reassoc double @llvm.vector.reduce.fmul.v8f64(double 1.000000e+00, <8 x double> %{{.*}})
  // LLVM-NINF-LABEL: define {{.*}} @test_mm512_reduce_mul_pd(
  // LLVM-NINF: call reassoc ninf {{.*}}double @llvm.vector.reduce.fmul.v8f64(

  // OGCG-LABEL: test_mm512_reduce_mul_pd
  // OGCG-NOT: reassoc
  // OGCG:    call reassoc {{.*}}double @llvm.vector.reduce.fmul.v8f64(double 1.000000e+00, <8 x double> %{{.*}})
  // OGCG-NOT: reassoc
  return _mm512_reduce_mul_pd(__W) * ExtraMulOp;
}


float test_mm512_reduce_add_ps(__m512 __W){
  // CIR-LABEL: test_mm512_reduce_add_ps
  // CIR: cir.call @_mm512_reduce_add_ps(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<16 x !cir.float>{{.*}}) -> !cir.float

  // CIR-LABEL: cir.func{{.*}} @_mm512_reduce_add_ps(
  // CIR: cir.vec.reduce(fadd, %[[V:.*]], %[[R:.*]]) : (!cir.vector<16 x !cir.float>, !cir.float) -> !cir.float {fastmath_flags = #cir.fastmath<reassoc>}

  // LLVM-LABEL: test_mm512_reduce_add_ps
  // LLVM: call reassoc float @llvm.vector.reduce.fadd.v16f32(float -0.000000e+00, <16 x float> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_add_ps
  // OGCG: call reassoc {{.*}}float @llvm.vector.reduce.fadd.v16f32(float -0.000000e+00, <16 x float> %{{.*}})
  return _mm512_reduce_add_ps(__W);
}

float test_mm512_reduce_mul_ps(__m512 __W){
  // CIR-LABEL: test_mm512_reduce_mul_ps
  // CIR: cir.call @_mm512_reduce_mul_ps(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<16 x !cir.float>{{.*}}) -> !cir.float

  // CIR-LABEL: cir.func{{.*}} @_mm512_reduce_mul_ps(
  // CIR: cir.vec.reduce(fmul, %[[V:.*]], %[[R:.*]]) : (!cir.vector<16 x !cir.float>, !cir.float) -> !cir.float {fastmath_flags = #cir.fastmath<reassoc>}

  // LLVM-LABEL: test_mm512_reduce_mul_ps
  // LLVM: call reassoc float @llvm.vector.reduce.fmul.v16f32(float 1.000000e+00, <16 x float> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_mul_ps
  // OGCG:    call reassoc {{.*}}float @llvm.vector.reduce.fmul.v16f32(float 1.000000e+00, <16 x float> %{{.*}})
  return _mm512_reduce_mul_ps(__W);
}
