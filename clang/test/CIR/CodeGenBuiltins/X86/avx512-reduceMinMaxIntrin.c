// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -fclangir -emit-cir -o - -Wall -Werror | FileCheck %s --check-prefixes=CIR
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -fclangir -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

double test_mm512_reduce_max_pd(__m512d __W, double ExtraAddOp){
  // CIR-LABEL: _mm512_reduce_max_pd
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmax" %[[V:.*]] : (!cir.vector<8 x !cir.double>) -> !cir.double

  // CIR-LABEL: test_mm512_reduce_max_pd
  // CIR: cir.call @_mm512_reduce_max_pd(%[[VEC:.*]]) : (!cir.vector<8 x !cir.double>) -> !cir.double

  // LLVM-LABEL: test_mm512_reduce_max_pd
  // LLVM: call double @llvm.vector.reduce.fmax.v8f64(<8 x double> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_max_pd
  // OGCG-NOT: nnan
  // OGCG: call nnan {{.*}}double @llvm.vector.reduce.fmax.v8f64(<8 x double> %{{.*}})
  // OGCG-NOT: nnan
  return _mm512_reduce_max_pd(__W) + ExtraAddOp;
}

double test_mm512_reduce_min_pd(__m512d __W, double ExtraMulOp){
  // CIR-LABEL: _mm512_reduce_min_pd
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmin" %[[V:.*]] : (!cir.vector<8 x !cir.double>) -> !cir.double

  // CIR-LABEL: test_mm512_reduce_min_pd
  // CIR: cir.call @_mm512_reduce_min_pd(%[[VEC:.*]]) : (!cir.vector<8 x !cir.double>) -> !cir.double

  // LLVM-LABEL: test_mm512_reduce_min_pd
  // LLVM: call double @llvm.vector.reduce.fmin.v8f64(<8 x double> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_min_pd
  // OGCG-NOT: nnan
  // OGCG:    call nnan {{.*}}double @llvm.vector.reduce.fmin.v8f64(<8 x double> %{{.*}})
  // OGCG-NOT: nnan
  return _mm512_reduce_min_pd(__W) * ExtraMulOp;
}

float test_mm512_reduce_max_ps(__m512 __W){
  // CIR-LABEL: _mm512_reduce_max_ps
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmax" %[[V:.*]] : (!cir.vector<16 x !cir.float>) -> !cir.float

  // CIR-LABEL: test_mm512_reduce_max_ps
  // CIR: cir.call @_mm512_reduce_max_ps(%[[VEC:.*]]) : (!cir.vector<16 x !cir.float>) -> !cir.float

  // LLVM-LABEL: test_mm512_reduce_max_ps
  // LLVM: call float @llvm.vector.reduce.fmax.v16f32(<16 x float> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_max_ps
  // OGCG: call nnan {{.*}}float @llvm.vector.reduce.fmax.v16f32(<16 x float> %{{.*}})
  return _mm512_reduce_max_ps(__W);
}

float test_mm512_reduce_min_ps(__m512 __W){
  // CIR-LABEL: _mm512_reduce_min_ps
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmin" %[[V:.*]] : (!cir.vector<16 x !cir.float>) -> !cir.float

  // CIR-LABEL: test_mm512_reduce_min_ps
  // CIR: cir.call @_mm512_reduce_min_ps(%[[VEC:.*]]) : (!cir.vector<16 x !cir.float>) -> !cir.float

  // LLVM-LABEL: test_mm512_reduce_min_ps
  // LLVM: call float @llvm.vector.reduce.fmin.v16f32(<16 x float> %{{.*}})

  // OGCG-LABEL: test_mm512_reduce_min_ps
  // OGCG: call nnan {{.*}}float @llvm.vector.reduce.fmin.v16f32(<16 x float> %{{.*}})
  return _mm512_reduce_min_ps(__W);
}
