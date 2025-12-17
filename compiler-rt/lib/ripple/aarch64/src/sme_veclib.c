#include <arm_sme.h>

typedef float v32f32 __attribute__((__vector_size__((128))))
__attribute__((aligned(16)));

#define RIPPLE_API_FIXED_VSCALE 4

// Verify if Ripple operation fixed vscale is same as runtime vscale.
bool verify_ripple_sme_api_fixed_vscale() {
  if ((svcntw() / sizeof(float)) == RIPPLE_API_FIXED_VSCALE)
    return true;
  else
    return false;
}

// Clear all ZA slices.
void ripple_sme_clearacc(void) __arm_streaming __arm_out("za") { svzero_za(); }

// Clear all ZA slices and set bias value.
void ripple_t32f32_sme_clearacc_set_bias(v32f32 bias) __arm_streaming __arm_out(
    "za") {
  svzero_za();

  // Load bias into two scalable vectors.
  svfloat32_t bias0 = *((svfloat32_t *)&bias);
  svfloat32_t bias1 = *((svfloat32_t *)((float *)&bias + svcntw()));

  // Broadcast 1.0f.
  svfloat32_t one = svdup_f32(1.0f);

  // Apply outer product to ZA tiles 0..3.
  svmopa_za32_m(0, svptrue_b32(), svptrue_b32(), one, bias0);
  svmopa_za32_m(1, svptrue_b32(), svptrue_b32(), one, bias1);
  svmopa_za32_m(2, svptrue_b32(), svptrue_b32(), one, bias0);
  svmopa_za32_m(3, svptrue_b32(), svptrue_b32(), one, bias1);
}

// Perform outer product.
void ripple_t1x32f32_t32f32_sme_outeracc(
    v32f32 A, v32f32 B) __arm_streaming __arm_inout("za") {
  svfloat32_t A0 = *((svfloat32_t *)&A);
  svfloat32_t A1 = *((svfloat32_t *)((float *)&A + svcntw()));
  svfloat32_t B0 = *((svfloat32_t *)&B);
  svfloat32_t B1 = *((svfloat32_t *)((float *)&B + svcntw()));

  // Outer product accumulate into ZA tile slices.
  svmopa_za32_m(0, svptrue_b32(), svptrue_b32(), A0, B0);
  svmopa_za32_m(1, svptrue_b32(), svptrue_b32(), A0, B1);
  svmopa_za32_m(2, svptrue_b32(), svptrue_b32(), A1, B0);
  svmopa_za32_m(3, svptrue_b32(), svptrue_b32(), A1, B1);
}

// Read data from ZA accumulator to memory.
void ripple_sme_readacc_f32(float *tile_base,
                            size_t row_stride) __arm_streaming __arm_in("za") {
  uint64_t tile2_offset = svcntw() * row_stride;

  for (uint64_t slice = 0; slice < svcntw(); ++slice) {
    // Read horizontal slices from ZA tiles.
    svfloat32_t r0 = svread_hor_za32_m(svundef_f32(), svptrue_b32(), 0, slice);
    svfloat32_t r1 = svread_hor_za32_m(svundef_f32(), svptrue_b32(), 1, slice);
    svfloat32_t r2 = svread_hor_za32_m(svundef_f32(), svptrue_b32(), 2, slice);
    svfloat32_t r3 = svread_hor_za32_m(svundef_f32(), svptrue_b32(), 3, slice);

    // Compute memory addresses.
    uint64_t slice_offset = slice * row_stride;
    float *addr0 = tile_base + slice_offset;
    float *addr1 = addr0 + svcntw();
    float *addr2 = addr0 + tile2_offset;
    float *addr3 = addr2 + svcntw();

    // Store vector slices.
    svst1_f32(svptrue_b32(), addr0, r0);
    svst1_f32(svptrue_b32(), addr1, r1);
    svst1_f32(svptrue_b32(), addr2, r2);
    svst1_f32(svptrue_b32(), addr3, r3);
  }
}

// Read data(clamped) from ZA accumulator to memory.
void ripple_sme_readacc_w_clamp_f32(
    float *tile_base, size_t row_stride, float clamp_min,
    float clamp_max) __arm_streaming __arm_in("za") {
  svfloat32_t vmin = svdup_f32(clamp_min);
  svfloat32_t vmax = svdup_f32(clamp_max);
  uint64_t tile2_offset = svcntw() * row_stride;

  for (uint64_t slice = 0; slice < svcntw(); ++slice) {
    // Read horizontal slices from ZA tiles.
    svfloat32_t r0 = svread_hor_za32_m(svundef_f32(), svptrue_b32(), 0, slice);
    svfloat32_t r1 = svread_hor_za32_m(svundef_f32(), svptrue_b32(), 1, slice);
    svfloat32_t r2 = svread_hor_za32_m(svundef_f32(), svptrue_b32(), 2, slice);
    svfloat32_t r3 = svread_hor_za32_m(svundef_f32(), svptrue_b32(), 3, slice);

    // Clamp each slice.
    r0 = svmax_f32_z(svptrue_b32(), svmin_f32_z(svptrue_b32(), r0, vmax), vmin);
    r1 = svmax_f32_z(svptrue_b32(), svmin_f32_z(svptrue_b32(), r1, vmax), vmin);
    r2 = svmax_f32_z(svptrue_b32(), svmin_f32_z(svptrue_b32(), r2, vmax), vmin);
    r3 = svmax_f32_z(svptrue_b32(), svmin_f32_z(svptrue_b32(), r3, vmax), vmin);

    // Compute memory addresses.
    uint64_t slice_offset = slice * row_stride;
    float *addr0 = tile_base + slice_offset;
    float *addr1 = addr0 + svcntw();
    float *addr2 = addr0 + tile2_offset;
    float *addr3 = addr2 + svcntw();

    // Store clamped vector slices.
    svst1_f32(svptrue_b32(), addr0, r0);
    svst1_f32(svptrue_b32(), addr1, r1);
    svst1_f32(svptrue_b32(), addr2, r2);
    svst1_f32(svptrue_b32(), addr3, r3);
  }
}
