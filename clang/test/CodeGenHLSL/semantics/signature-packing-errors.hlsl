// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=prefix-stable -DSTACKED -o /dev/null -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=optimized -DSTACKED -o /dev/null -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=prefix-stable -DOUTPUT -o /dev/null -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=optimized -DOUTPUT -o /dev/null -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=prefix-stable -o /dev/null -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=optimized -o /dev/null -verify %s

// Packing failures must be diagnosed rather than emitting unallocated metadata
// or reporting a fatal LLVM error. Stacked inputs are limited to 32 rows too.
#ifdef STACKED
[shader("vertex")]
void stacked_overflow(float data[33] : A) {} // expected-error {{failed to pack input signature: signature elements do not fit in 32 rows (element 0)}}

#elif defined(OUTPUT)
struct Output {
  float4 data[33] : A;
};

[shader("vertex")]
Output output_overflow() { // expected-error {{failed to pack output signature: signature elements do not fit in 32 rows (element 0)}}
  Output output;
  return output;
}

#else
[shader("pixel")]
void input_overflow(float4 data[33] : A) {} // expected-error {{failed to pack input signature: signature elements do not fit in 32 rows (element 0)}}
#endif
