// CPU-side compilation on x86 (no errors expected).
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64 -x cuda -fsyntax-only -verify=cpu %s

// GPU-side compilation on x86 (no errors expected)
// RUN: %clang_cc1 -triple nvptx64 -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -x cuda -fsyntax-only -verify=gpu %s

// cpu-no-diagnostics
typedef _Complex float __cfloat128 __attribute__ ((__mode__ (__TC__)));
typedef __float128 _Float128;

// gpu-note@+1 {{'a' defined here}}
__attribute__((device)) __float128 f(__float128 a, float b) {
    // gpu-note@+1 {{'c' defined here}}
  __float128 c = b + 1.0;
  // gpu-error@+2 {{'a' requires 128 bit size '__float128' type support, but target 'nvptx64' does not support it}}
  // gpu-error@+1 {{'c' requires 128 bit size '__float128' type support, but target 'nvptx64' does not support it}}
  return a + c;
}