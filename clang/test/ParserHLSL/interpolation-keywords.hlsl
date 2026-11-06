// RUN: %clang_cc1 -x c++ -verify %s
// RUN: %clang_cc1 -x c -verify %s
// expected-no-diagnostics

// Interpolation modifiers are keywords only in HLSL.
int nointerpolation, linear, centroid, noperspective, sample, center;
