// CPU-side compilation on x86 (no errors expected).
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64 -x cuda -fsyntax-only -verify %s

// GPU-side compilation on x86 (no errors expected)
// RUN: %clang_cc1 -triple nvptx64 -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -x cuda -fsyntax-only -verify %s

// expected-no-diagnostics
typedef _Complex float __cfloat128 __attribute__ ((__mode__ (__TC__)));
typedef __float128 _Float128;
