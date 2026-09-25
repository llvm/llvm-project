// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -fclangir -emit-cir -o - -Wall -Werror | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -fclangir -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -menable-no-infs -fclangir -emit-cir -o - -Wall -Werror | FileCheck %s --check-prefix=CIR-NINF
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -menable-no-infs -fclangir -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=LLVM-NINF
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -menable-no-infs -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=LLVM-NINF

#include <immintrin.h>

double test_mm512_reduce_max_pd(__m512d __W, double ExtraAddOp){
  // CIR-LABEL: test_mm512_reduce_max_pd
  // CIR: cir.call @_mm512_reduce_max_pd(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<8 x !cir.double>{{.*}}) -> !cir.double

  // CIR-LABEL: cir.func{{.*}} @_mm512_reduce_max_pd(
  // CIR: cir.vec.reduce(fmax, %[[V:.*]]) : (!cir.vector<8 x !cir.double>) -> !cir.double {fastmath_flags = #cir.fastmath<nnan>}
  // CIR-NINF-LABEL: cir.func{{.*}} @_mm512_reduce_max_pd(
  // CIR-NINF: cir.vec.reduce(fmax, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<nnan, ninf>}

  // LLVM-LABEL: test_mm512_reduce_max_pd
  // LLVM: call nnan double @llvm.vector.reduce.fmax.v8f64(<8 x double> %{{.*}})
  // LLVM-NINF-LABEL: define {{.*}} @test_mm512_reduce_max_pd(
  // LLVM-NINF: call nnan ninf {{.*}}double @llvm.vector.reduce.fmax.v8f64(

  // OGCG-LABEL: test_mm512_reduce_max_pd
  // OGCG-NOT: nnan
  // OGCG: call nnan {{.*}}double @llvm.vector.reduce.fmax.v8f64(<8 x double> %{{.*}})
  // OGCG-NOT: nnan
  return _mm512_reduce_max_pd(__W) + ExtraAddOp;
}

double test_mm512_reduce_min_pd(__m512d __W, double ExtraMulOp){
  // CIR-LABEL: test_mm512_reduce_min_pd
  // CIR: cir.call @_mm512_reduce_min_pd(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<8 x !cir.double>{{.*}}) -> !cir.double

  // CIR-LABEL: cir.func{{.*}} @_mm512_reduce_min_pd(
  // CIR: cir.vec.reduce(fmin, %[[V:.*]]) : (!cir.vector<8 x !cir.double>) -> !cir.double {fastmath_flags = #cir.fastmath<nnan>}
  // CIR-NINF-LABEL: cir.func{{.*}} @_mm512_reduce_min_pd(
  // CIR-NINF: cir.vec.reduce(fmin, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<nnan, ninf>}

  // LLVM-LABEL: test_mm512_reduce_min_pd
  // LLVM: call nnan double @llvm.vector.reduce.fmin.v8f64(<8 x double> %{{.*}})
  // LLVM-NINF-LABEL: define {{.*}} @test_mm512_reduce_min_pd(
  // LLVM-NINF: call nnan ninf {{.*}}double @llvm.vector.reduce.fmin.v8f64(

  // OGCG-LABEL: test_mm512_reduce_min_pd
  // OGCG-NOT: nnan
  // OGCG:    call nnan {{.*}}double @llvm.vector.reduce.fmin.v8f64(<8 x double> %{{.*}})
  // OGCG-NOT: nnan
  return _mm512_reduce_min_pd(__W) * ExtraMulOp;
}

float test_mm512_reduce_max_ps(__m512 __W){
  // CIR-LABEL: test_mm512_reduce_max_ps
  // CIR: cir.call @_mm512_reduce_max_ps(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<16 x !cir.float>{{.*}}) -> !cir.float

  // CIR-LABEL: cir.func{{.*}} @_mm512_reduce_max_ps(
  // CIR: cir.vec.reduce(fmax, %[[V:.*]]) : (!cir.vector<16 x !cir.float>) -> !cir.float {fastmath_flags = #cir.fastmath<nnan>}

  // LLVM-LABEL: test_mm512_reduce_max_ps
  // LLVM: call nnan float @llvm.vector.reduce.fmax.v16f32(<16 x float> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_max_ps
  // OGCG: call nnan {{.*}}float @llvm.vector.reduce.fmax.v16f32(<16 x float> %{{.*}})
  return _mm512_reduce_max_ps(__W);
}

float test_mm512_reduce_min_ps(__m512 __W){
  // CIR-LABEL: test_mm512_reduce_min_ps
  // CIR: cir.call @_mm512_reduce_min_ps(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<16 x !cir.float>{{.*}}) -> !cir.float

  // CIR-LABEL: cir.func{{.*}} @_mm512_reduce_min_ps(
  // CIR: cir.vec.reduce(fmin, %[[V:.*]]) : (!cir.vector<16 x !cir.float>) -> !cir.float {fastmath_flags = #cir.fastmath<nnan>}

  // LLVM-LABEL: test_mm512_reduce_min_ps
  // LLVM: call nnan float @llvm.vector.reduce.fmin.v16f32(<16 x float> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_min_ps
  // OGCG: call nnan {{.*}}float @llvm.vector.reduce.fmin.v16f32(<16 x float> %{{.*}})
  return _mm512_reduce_min_ps(__W);
}
