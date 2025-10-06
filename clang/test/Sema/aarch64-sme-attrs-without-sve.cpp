// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

void test_streaming(svint32_t *out, svint32_t *in) __arm_streaming {
  *out = *in;
}

void test_non_streaming(svint32_t *out, svint32_t *in) {
  *out = *in; // expected-error {{SVE vector type 'svint32_t' (aka '__SVInt32_t') cannot be used in a non-streaming function}} \
                 expected-error {{SVE vector type 'svint32_t' (aka '__SVInt32_t') cannot be used in a non-streaming function}}
}

// This previously led to a diagnostic that '&a' could not be used in a non-streaming function,
// even though all functions are streaming.
void test_both_streaming(int32_t *out) __arm_streaming {
  svint32_t a;
  [&a, &out]() __arm_streaming {
    a = svdup_s32(1);
    svst1(svptrue_b32(), out, a);
  }();
}

void test_lambda_streaming(int32_t *out) {
  svint32_t a; // expected-error {{SVE vector type 'svint32_t' (aka '__SVInt32_t') cannot be used in a non-streaming function}}
  [&a, &out]() __arm_streaming {
    a = 1;
    svst1(svptrue_b32(), out, a);
  }();
}

void test_lambda_non_streaming_capture_do_nothing() __arm_streaming {
  svint32_t a;
  [&a] {
    // Do nothing.
  }();
}

// Error: Non-streaming function attempts to dereference capture:
void test_lambda_non_streaming_capture_return_vector() __arm_streaming {
  svint32_t a;
  [&a] {
    return a; // expected-error {{SVE vector type 'svint32_t' (aka '__SVInt32_t') cannot be used in a non-streaming function}}
  }();
}

// By reference capture, only records and uses the address of `a`:
// FIXME: This should be okay.
void test_lambda_non_streaming_capture_return_address() __arm_streaming {
  svint32_t a;
  [&a] {
    return &a; // expected-error {{SVE vector type 'svint32_t' (aka '__SVInt32_t') cannot be used in a non-streaming function}}
  }();
}
