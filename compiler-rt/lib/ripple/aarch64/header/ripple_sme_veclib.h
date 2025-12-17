#pragma once
#include <ripple.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool verify_ripple_sme_api_fixed_vscale();

void sme_clearacc(ripple_block_t) __arm_streaming __arm_out("za");

void sme_clearacc_set_bias(ripple_block_t,
                           /*bias*/ float) __arm_streaming __arm_out("za");

void sme_outeracc(
    /*vertical_vector*/ float,
    /*horizontal_vector*/ float) __arm_streaming __arm_inout("za");

void sme_readacc_f32(ripple_block_t, /*tile_base*/ float *,
                     /*row_stride*/ size_t) __arm_streaming __arm_in("za");

void sme_readacc_w_clamp_f32(
    ripple_block_t, /*tile_base*/ float *,
    /*row_stride*/ size_t, /*clamp_min*/ float,
    /*clamp_max*/ float) __arm_streaming __arm_in("za");

#ifdef __cplusplus
}
#endif
