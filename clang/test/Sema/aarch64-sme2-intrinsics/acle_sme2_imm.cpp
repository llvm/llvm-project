// RUN: %clang_cc1 -triple aarch64-none-linux-gnu \
// RUN:    -target-feature +sve2 -target-feature +sme2 -target-feature +sme-i16i64 -target-feature +sme-f64f64 -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

#include <arm_sme_draft_spec_subject_to_change.h>

void test_ldr_str_zt(const void *const_base, void *base) __arm_streaming_compatible __arm_shared_za __arm_preserves_za {
  svldr_zt(1, const_base); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svstr_zt(1, base);       // expected-error {{argument value 1 is outside the valid range [0, 0]}}
}
