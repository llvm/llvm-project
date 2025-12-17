// REQUIRES: aarch64-registered-target

// RUN: %clang -march=armv9.2-a+sme --target=aarch64-unknown-linux-gnu -c -O2 -msve-streaming-vector-bits=512 -emit-llvm %s -DCOMPILE_LIB -o %t.bc
// RUN: %clang -march=armv9.2-a+sme --target=aarch64-unknown-linux-gnu -O2 -fno-unroll-loops -msve-streaming-vector-bits=512 -fenable-ripple -fripple-lib=%t.bc -mllvm -ripple-disable-link %s -emit-llvm -S -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

// Unfail test once ripple accept external functions with block shape arguments.
// XFAIL: *

#ifdef COMPILE_LIB
#include <arm_sme.h>

typedef float v32f32 __attribute__((__vector_size__((128))))
__attribute__((aligned(16)));

void ripple_t32f32_sme_clearacc_set_bias(v32f32 bias) __arm_streaming __arm_out("za") {
  // Dummy implementation.
  svzero_za();
}

void ripple_t1x32f32_t32f32_sme_outeracc(
    v32f32 A, v32f32 B) __arm_streaming __arm_inout("za") {
  // Dummy implementation.
  svfloat32_t A0 = *((svfloat32_t *)&A);
  svfloat32_t B0 = *((svfloat32_t *)&B);
  svmopa_za32_m(0, svptrue_b32(), svptrue_b32(), A0, B0);
}

void ripple_sme_readacc_w_clamp_f32(
    float *tile_base, size_t row_stride, float clamp_min,
    float clamp_max) __arm_streaming __arm_in("za") {
  // Dummy implementation.
  svfloat32_t r0 = svread_hor_za32_m(svundef_f32(), svptrue_b32(), 0, 0);
  svst1_f32(svptrue_b32(), tile_base, r0);
}

#else
#include <ripple.h>

#define SME_SIZE 32

extern void sme_clearacc_set_bias(ripple_block_t, /*bias*/ float) __arm_streaming __arm_out("za");
extern void sme_outeracc(/*vertical_vector*/ float,
                         /*horizontal_vector*/ float) __arm_streaming __arm_inout("za");
extern void sme_readacc_w_clamp_f32(ripple_block_t, /*tile_base*/ float *, /*row_stride*/ size_t,
                                    /*clamp_min*/ float, /*clamp_max*/ float) __arm_streaming __arm_in("za");

// Check SME external functions operating on expected tensor shape are used to
// widen corresponding scalar calls.
//
// CHECK-LABEL: define {{.*}}void @ripple_matmul_f32
// CHECK-DAG: call void @ripple_t32f32_sme_clearacc_set_bias
// CHECK-DAG: call void @ripple_sme_readacc_w_clamp_f32
// CHECK-DAG: call void @ripple_t1x32f32_t32f32_sme_outeracc

__arm_locally_streaming __arm_new("za") void ripple_matmul_f32(
    size_t M, size_t N, size_t K, float *restrict A_packed,
    float *restrict B_packed, float *restrict C, float ClampMin,
    float ClampMax) {
  __builtin_assume(M % SME_SIZE == 0);
  __builtin_assume(N % SME_SIZE == 0);
  ripple_block_t sme_block = ripple_set_block_shape(0, SME_SIZE, SME_SIZE);
  size_t x = ripple_id(sme_block, 0);
  size_t y = ripple_id(sme_block, 1);
  float *A_packed_ptr = A_packed;
  float *A_packed_ptr_per_row = A_packed;
  float *B_packed_ptr = B_packed;

  unsigned TileOffset = 0;
  ripple_parallel_full(sme_block, 1);
  for (size_t i = 0; i < M; ++i) {
    ripple_parallel_full(sme_block, 0);
    for (size_t j = 0; j < N; ++j) {
      sme_clearacc_set_bias(sme_block, B_packed_ptr[x]);
      A_packed_ptr = A_packed_ptr_per_row;
      B_packed_ptr += SME_SIZE;
      for (size_t k = 0; k < K; ++k) {
        sme_outeracc(A_packed_ptr[y], B_packed_ptr[x]);
        A_packed_ptr += SME_SIZE;
        B_packed_ptr += SME_SIZE;
      }
      sme_readacc_w_clamp_f32(sme_block, C + TileOffset, N, ClampMin, ClampMax);
      TileOffset += SME_SIZE;
    }
    A_packed_ptr_per_row = A_packed_ptr;
    B_packed_ptr = B_packed;
    TileOffset += (SME_SIZE - 1) * N;
  }
}
#endif
