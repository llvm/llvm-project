// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512fp16 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512fp16 -fclangir -emit-llvm -o %t.ll  -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
#include <immintrin.h>

_Float16 test_mm256_reduce_add_ph(__m256h __W) {
  // CIR-LABEL: _mm256_reduce_add_ph
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fadd" %[[R:.*]], %[[V:.*]] : (!cir.f16, !cir.vector<16 x !cir.f16>) -> !cir.f16

  // CIR-LABEL: test_mm256_reduce_add_ph
  // CIR: cir.call @_mm256_reduce_add_ph(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<16 x !cir.f16>) -> !cir.f16

  // LLVM-LABEL: test_mm256_reduce_add_ph
  // LLVM: call half @llvm.vector.reduce.fadd.v16f16(half 0xH8000, <16 x half> %{{.*}})

  // OGCG-LABEL: test_mm256_reduce_add_ph
  // OGCG: call reassoc {{.*}}@llvm.vector.reduce.fadd.v16f16(half 0xH8000, <16 x half> %{{.*}})
  return _mm256_reduce_add_ph(__W);
}

_Float16 test_mm256_reduce_mul_ph(__m256h __W) {
  // CIR-LABEL: _mm256_reduce_mul_ph
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmul" %[[R:.*]], %[[V:.*]] : (!cir.f16, !cir.vector<16 x !cir.f16>) -> !cir.f16

  // CIR-LABEL: test_mm256_reduce_mul_ph
  // CIR: cir.call @_mm256_reduce_mul_ph(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<16 x !cir.f16>) -> !cir.f16

  // LLVM-LABEL: test_mm256_reduce_mul_ph
  // LLVM: call half @llvm.vector.reduce.fmul.v16f16(half 0xH3C00, <16 x half> %{{.*}})

  // OGCG-LABEL: test_mm256_reduce_mul_ph
  // OGCG: call reassoc {{.*}}@llvm.vector.reduce.fmul.v16f16(half 0xH3C00, <16 x half> %{{.*}})
  return _mm256_reduce_mul_ph(__W);
}

_Float16 test_mm256_reduce_max_ph(__m256h __W) {
  // CIR-LABEL: _mm256_reduce_max_ph
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmax" %[[V:.*]] (!cir.vector<16 x !cir.f16>) -> !cir.f16

  // CIR-LABEL: test_mm256_reduce_max_ph
  // CIR: cir.call @_mm256_reduce_max_ph(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<16 x !cir.f16>) -> !cir.f16

  // LLVM-LABEL: test_mm256_reduce_max_ph
  // LLVM: call half @llvm.vector.reduce.fmax.v16f16(<16 x half> %{{.*}})

  // OGCG-LABEL: test_mm256_reduce_max_ph
  // OGCG: call nnan {{.*}}@llvm.vector.reduce.fmax.v16f16(<16 x half> %{{.*}})
  return _mm256_reduce_max_ph(__W);
}

_Float16 test_mm256_reduce_min_ph(__m256h __W) {
  // CIR-LABEL: _mm256_reduce_min_ph
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmin" %[[V:.*]] : (!cir.vector<16 x !cir.f16>) -> !cir.f16

  // CIR-LABEL: test_mm256_reduce_min_ph
  // CIR: cir.call @_mm256_reduce_min_ph(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<16 x !cir.f16>) -> !cir.f16

  // LLVM-LABEL: test_mm256_reduce_min_ph
  // LLVM: call half @llvm.vector.reduce.fmin.v16f16(<16 x half> %{{.*}})

  // OGCG-LABEL: test_mm256_reduce_min_ph
  // OGCG: call nnan {{.*}}@llvm.vector.reduce.fmin.v16f16(<16 x half> %{{.*}})
  return _mm256_reduce_min_ph(__W);
}

_Float16 test_mm_reduce_add_ph(__m128h __W) {
  // CIR-LABEL: _mm_reduce_add_ph
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fadd" %[[R:.*]], %[[V:.*]] : (!cir.f16, !cir.vector<8 x !cir.f16>) -> !cir.f16

  // CIR-LABEL: test_mm_reduce_add_ph
  // CIR: cir.call @_mm_reduce_add_ph(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<8 x !cir.f16>) -> !cir.f16

  // LLVM-LABEL: test_mm_reduce_add_ph
  // LLVM: call half @llvm.vector.reduce.fadd.v8f16(half 0xH8000, <8 x half> %{{.*}})

  // OGCG-LABEL: test_mm_reduce_add_ph
  // OGCG: call reassoc {{.*}}@llvm.vector.reduce.fadd.v8f16(half 0xH8000, <8 x half> %{{.*}})
  return _mm_reduce_add_ph(__W);
}

_Float16 test_mm_reduce_mul_ph(__m128h __W) {
  // CIR-LABEL: _mm_reduce_mul_ph
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmul" %[[R:.*]], %[[V:.*]] : (!cir.f16, !cir.vector<8 x !cir.f16>) -> !cir.f16

  // CIR-LABEL: test_mm_reduce_mul_ph
  // CIR: cir.call @_mm_reduce_mul_ph(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<8 x !cir.f16>) -> !cir.f16

  // LLVM-LABEL: test_mm_reduce_mul_ph
  // LLVM: call half @llvm.vector.reduce.fmul.v8f16(half 0xH3C00, <8 x half> %{{.*}})

  // OGCG-LABEL: test_mm_reduce_mul_ph
  // OGCG: call reassoc {{.*}}@llvm.vector.reduce.fmul.v8f16(half 0xH3C00, <8 x half> %{{.*}})
  return _mm_reduce_mul_ph(__W);
}

_Float16 test_mm_reduce_max_ph(__m128h __W) {
  // CIR-LABEL: _mm_reduce_max_ph
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmax" %[[V:.*]] (!cir.vector<8 x !cir.f16>) -> !cir.f16

  // CIR-LABEL: test_mm_reduce_max_ph
  // CIR: cir.call @_mm_reduce_max_ph(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<8 x !cir.f16>) -> !cir.f16

  // LLVM-LABEL: test_mm_reduce_max_ph
  // LLVM: call half @llvm.vector.reduce.fmax.v8f16(<8 x half> %{{.*}})

  // OGCG-LABEL: test_mm_reduce_max_ph
  // OGCG: call nnan {{.*}}@llvm.vector.reduce.fmax.v8f16(<8 x half> %{{.*}})
  return _mm_reduce_max_ph(__W);
}

_Float16 test_mm_reduce_min_ph(__m128h __W) {
  // CIR-LABEL: _mm_reduce_min_ph
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmin" %[[V:.*]] : (!cir.vector<8 x !cir.f16>) -> !cir.f16

  // CIR-LABEL: test_mm_reduce_min_ph
  // CIR: cir.call @_mm_reduce_min_ph(%[[VEC:.*]]) {nobuiltin, nobuiltins = [{{.*}}]} : (!cir.vector<8 x !cir.f16>) -> !cir.f16

  // LLVM-LABEL: test_mm_reduce_min_ph
  // LLVM: call half @llvm.vector.reduce.fmin.v8f16(<8 x half> %{{.*}})

  // OGCG-LABEL: test_mm_reduce_min_ph
  // OGCG: call nnan {{.*}}@llvm.vector.reduce.fmin.v8f16(<8 x half> %{{.*}})
  return _mm_reduce_min_ph(__W);
}

