// RUN: %clang_cc1  -x hlsl -triple dxil--shadermodel6.7-library  %s  -verify

// Make sure WaveActiveCountBits is accepted.

// expected-no-diagnostics
unsigned foo(bool b) {
    return WaveActiveCountBits(b);
}
