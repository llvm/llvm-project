// Case 1: No vscale flags — should only produce warnings
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +bf16 -target-feature +sme -target-feature +sme2 -target-feature +sve -Waarch64-sme-attributes -fsyntax-only -verify=expected-noflags %s

// Case 2: Explicit mismatch in vscale flags — should produce errors for 
// streaming and non-streaming callers
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +bf16 -target-feature +sme -target-feature +sme2 -target-feature +sve -Waarch64-sme-attributes -fsyntax-only -mvscale-min=1 -mvscale-max=1 -mvscale-streaming-min=2 -mvscale-streaming-max=2 -verify=expected-flags %s

void sme_streaming_with_vl_arg(__SVInt8_t a) __arm_streaming;

__SVInt8_t sme_streaming_returns_vl(void) __arm_streaming;

void sme_streaming_compatible_with_vl_arg(__SVInt8_t a) __arm_streaming_compatible;

__SVInt8_t sme_streaming_compatible_returns_vl(void) __arm_streaming_compatible;

void sme_no_streaming_with_vl_arg(__SVInt8_t a);

__SVInt8_t sme_no_streaming_returns_vl(void);


void sme_no_streaming_calling_streaming_with_vl_args() {
  __SVInt8_t a;
  // expected-noflags-warning@+2 {{passing a VL-dependent argument to a function with a different streaming-mode is undefined behaviour when the streaming and non-streaming vector lengths are different at runtime}}
  // expected-flags-error@+1 {{passing a VL-dependent argument to a function with a different streaming-mode is undefined behaviour because the streaming vector length (256 bit) and non-streaming vector length (128 bit) differ}}
  sme_streaming_with_vl_arg(a);
}

void sme_no_streaming_calling_streaming_with_return_vl() {
  // expected-noflags-warning@+2 {{returning a VL-dependent argument from a function with a different streaming-mode is undefined behaviour when the streaming and non-streaming vector lengths are different at runtime}}
  // expected-flags-error@+1 {{returning a VL-dependent argument from a function with a different streaming-mode is undefined behaviour because the streaming vector length (256 bit) and non-streaming vector length (128 bit) differ}}
  __SVInt8_t r = sme_streaming_returns_vl();
}

void sme_streaming_calling_non_streaming_with_vl_args(void) __arm_streaming {
  __SVInt8_t a;
  // expected-noflags-warning@+2 {{passing a VL-dependent argument to a function with a different streaming-mode is undefined behaviour when the streaming and non-streaming vector lengths are different at runtime}}
  // expected-flags-error@+1 {{passing a VL-dependent argument to a function with a different streaming-mode is undefined behaviour because the streaming vector length (256 bit) and non-streaming vector length (128 bit) differ}}
  sme_no_streaming_with_vl_arg(a);
}

void sme_streaming_calling_non_streaming_with_return_vl(void) __arm_streaming {
  // expected-noflags-warning@+2 {{returning a VL-dependent argument from a function with a different streaming-mode is undefined behaviour when the streaming and non-streaming vector lengths are different at runtime}}
  // expected-flags-error@+1 {{returning a VL-dependent argument from a function with a different streaming-mode is undefined behaviour because the streaming vector length (256 bit) and non-streaming vector length (128 bit) differ}}
  __SVInt8_t r = sme_no_streaming_returns_vl();
}

void sme_streaming_compatible_calling_streaming_with_vl_args(__SVInt8_t arg) __arm_streaming_compatible {
  // expected-noflags-warning@+2 {{passing a VL-dependent argument to a function with a different streaming-mode is undefined behaviour when the streaming and non-streaming vector lengths are different at runtime}}
  // expected-flags-warning@+1 {{passing a VL-dependent argument to a streaming function is undefined behaviour when the streaming-compatible caller is not in streaming mode, because the streaming vector length (256 bit) and non-streaming vector length (128 bit) differ}}
  sme_streaming_with_vl_arg(arg);
}

void sme_streaming_compatible_calling_sme_streaming_return_vl(void) __arm_streaming_compatible {
  // expected-noflags-warning@+2 {{returning a VL-dependent argument from a function with a different streaming-mode is undefined behaviour when the streaming and non-streaming vector lengths are different at runtime}}
  // expected-flags-warning@+1 {{returning a VL-dependent argument from a streaming function is undefined behaviour when the streaming-compatible caller is not in streaming mode, because the streaming vector length (256 bit) and non-streaming vector length (128 bit) differ}}
  __SVInt8_t r = sme_streaming_returns_vl();
}

void sme_streaming_compatible_calling_no_streaming_with_vl_args(__SVInt8_t arg) __arm_streaming_compatible {
  // expected-noflags-warning@+2 {{passing a VL-dependent argument to a function with a different streaming-mode is undefined behaviour when the streaming and non-streaming vector lengths are different at runtime}}
  // expected-flags-warning@+1 {{passing a VL-dependent argument to a non-streaming function is undefined behaviour when the streaming-compatible caller is in streaming mode, because the streaming vector length (256 bit) and non-streaming vector length (128 bit) differ}}
  sme_no_streaming_with_vl_arg(arg);
}

void sme_streaming_compatible_calling_no_sme_streaming_return_vl(void) __arm_streaming_compatible {
  // expected-noflags-warning@+2 {{returning a VL-dependent argument from a function with a different streaming-mode is undefined behaviour when the streaming and non-streaming vector lengths are different at runtime}}
  // expected-flags-warning@+1 {{returning a VL-dependent argument from a non-streaming function is undefined behaviour when the streaming-compatible caller is in streaming mode, because the streaming vector length (256 bit) and non-streaming vector length (128 bit) differ}}
  __SVInt8_t r = sme_no_streaming_returns_vl();
}

void sme_streaming_calling_streaming_with_vl_args(__SVInt8_t a) __arm_streaming {
  sme_streaming_with_vl_arg(a);
}

void sme_streaming_calling_streaming_with_return_vl(void) __arm_streaming {
  __SVInt8_t r = sme_streaming_returns_vl();
}

void sme_streaming_calling_streaming_compatible_with_vl_args(__SVInt8_t a) __arm_streaming {
  sme_streaming_compatible_with_vl_arg(a);
}

void sme_streaming_calling_streaming_compatible_with_return_vl(void) __arm_streaming {
  __SVInt8_t r = sme_streaming_compatible_returns_vl();
}

void sme_no_streaming_calling_streaming_compatible_with_vl_args() {
  __SVInt8_t a;
  sme_streaming_compatible_with_vl_arg(a);
}

void sme_no_streaming_calling_streaming_compatible_with_return_vl() {
  __SVInt8_t r = sme_streaming_compatible_returns_vl();
}

void sme_no_streaming_calling_non_streaming_with_vl_args() {
  __SVInt8_t a;
  sme_no_streaming_with_vl_arg(a);
}

void sme_no_streaming_calling_non_streaming_with_return_vl() {
  __SVInt8_t r = sme_no_streaming_returns_vl();
}

void sme_streaming_compatible_calling_streaming_compatible_with_vl_args(__SVInt8_t arg) __arm_streaming_compatible {
  sme_streaming_compatible_with_vl_arg(arg);
}

void sme_streaming_compatible_calling_streaming_compatible_with_return_vl(void) __arm_streaming_compatible {
  __SVInt8_t r = sme_streaming_compatible_returns_vl();
}
