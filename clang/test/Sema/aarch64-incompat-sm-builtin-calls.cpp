// RUN: %clang_cc1  -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:   -target-feature +sme -target-feature +neon -x c++ -std=c++20 -Waarch64-sme-attributes -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>
#include <arm_neon.h>

void use_streaming_builtin_in_lambda(uint32_t slice_base, svbool_t pg, const void *ptr) __arm_streaming __arm_out("za")
{
  [&]{
    /// The lambda is its own function and does not inherit the SME attributes (so this should error).
    // expected-error@+1 {{builtin can only be called from a streaming function}}
    svld1_hor_za64(0, slice_base, pg, ptr);
  }();
}

void use_streaming_builtin(uint32_t slice_base, svbool_t pg, const void *ptr) __arm_streaming __arm_out("za")
{
  /// Without the lambda the same builtin is okay (as the SME attributes apply).
  svld1_hor_za64(0, slice_base, pg, ptr);
}

int16x8_t use_neon_builtin_sm(int16x8_t splat) __arm_streaming_compatible {
  // expected-error@+1 {{builtin can only be called from a non-streaming function}}
  return (int16x8_t)__builtin_neon_vqaddq_v((int8x16_t)splat, (int8x16_t)splat, 33);
}

int16x8_t use_neon_builtin_sm_in_lambda(int16x8_t splat) __arm_streaming_compatible {
  return [&]{
    /// This should not error (as we switch out of streaming mode to execute the lambda).
    /// Note: The result int16x8_t is spilled and reloaded as a q-register.
    return (int16x8_t)__builtin_neon_vqaddq_v((int8x16_t)splat, (int8x16_t)splat, 33);
  }();
}

float use_incomp_sve_builtin_sm() __arm_streaming {
  // expected-error@+1 {{builtin can only be called from a non-streaming function}}
  return svadda(svptrue_b32(), 0, svdup_f32(1));
}

float incomp_sve_sm_fadda_sm_in_lambda(void) __arm_streaming {
  return [&]{
    /// This should work like the Neon builtin.
    return svadda(svptrue_b32(), 0, svdup_f32(1));
  }();
}
