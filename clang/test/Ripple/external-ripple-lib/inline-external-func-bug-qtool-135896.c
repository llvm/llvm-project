// REQUIRES: aarch64-registered-target

// RUN: %clang -march=armv9.2-a+sme --target=aarch64-unknown-linux-gnu -c -O2 -msve-streaming-vector-bits=512 -emit-llvm %s -DCOMPILE_LIB -o %t.bc
// RUN: %clang -march=armv9.2-a+sme --target=aarch64-unknown-linux-gnu -O2 -msve-streaming-vector-bits=512 -fenable-ripple -fripple-lib=%t.bc %s -emit-llvm -S -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

#ifdef COMPILE_LIB
#include <arm_sme.h>

typedef float v32f32 __attribute__((__vector_size__((128))))
__attribute__((aligned(16)));
typedef float v1024f32 __attribute__((__vector_size__((4096))))
__attribute__((aligned(16)));

v1024f32 ripple_ret_t32x32f32_t32x32f32_t1x32f32_t32f32_sme_outeracc(
    v1024f32 ZA, v32f32 A, v32f32 B) __arm_streaming __arm_inout("za") {
  svfloat32_t A0 = *((svfloat32_t *)&A);
  svfloat32_t B0 = *((svfloat32_t *)&B);

  svmopa_za32_m(0, svptrue_b32(), svptrue_b32(), A0, B0);

  return ZA;
}

void ripple_t32x32f32_sme_readacc_w_clamp(
    v1024f32 ZA, float *tile_base, uint64_t row_stride, float clamp_min,
    float clamp_max) __arm_streaming __arm_in("za") {
  svfloat32_t vmin = svdup_f32(clamp_min);
  svfloat32_t vmax = svdup_f32(clamp_max);

  #pragma clang loop unroll(disable)
  for (uint64_t slice = 0; slice < svcntw(); ++slice) {
    svfloat32_t r0 = svread_hor_za32_m(svundef_f32(), svptrue_b32(), 0, slice);

    r0 = svmax_f32_z(svptrue_b32(), svmin_f32_z(svptrue_b32(), r0, vmax), vmin);

    uint64_t slice_offset = slice * row_stride;
    float *addr0 = tile_base + slice_offset;
    svst1_f32(svptrue_b32(), addr0, r0);
  }
}
#else
#include <stddef.h>
#include <stdint.h>

void ripple_t32x32f32_sme_readacc_w_clamp(void *arg1, void *arg2, int64_t N, float ClampMin, float ClampMax);
void ripple_ret_t32x32f32_t32x32f32_t1x32f32_t32f32_sme_outeracc(void *arg1, void *arg2, void *A_packed, void *B_packed);

// Check that all external functions are inlined.
// CHECK-NOT: tail call void @ripple_t32x32f32_sme_readacc_w_clamp
// CHECK-NOT: tail call void @ripple_ret_t32x32f32_t32x32f32_t1x32f32_t32f32_sme_outeracc

// Test function
void test(void *A_packed, void *B_packed, int64_t N, float ClampMin, float ClampMax) {
    ripple_t32x32f32_sme_readacc_w_clamp(NULL, NULL, N, ClampMin, ClampMax);
    ripple_ret_t32x32f32_t32x32f32_t1x32f32_t32f32_sme_outeracc(NULL, NULL, A_packed, B_packed);
}

int main() {
    int64_t N = 1024;
    float ClampMin = 0.0f, ClampMax = 1.0f;

    // Dummy packed arrays
    float A[32 * 32], B[32 * 32];
    test(A, B, N, ClampMin, ClampMax);

    return 0;
}
#endif
