// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=prefix-stable -o /dev/null -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=optimized -o /dev/null -verify %s

struct Output {
  float4 data[33] : A;
};

[shader("vertex")]
// expected-error@+1 {{failed to pack output signature: signature elements do not fit in 32 rows (element 0)}}
Output output_overflow() {
  Output output;
  return output;
}
