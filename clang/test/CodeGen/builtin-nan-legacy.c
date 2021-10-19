// RUN: %clang -target mipsel-unknown-linux -mnan=legacy -emit-llvm -S %s -o - | FileCheck %s
// CHECK: float 0x7FFC000000000000, float 0x7FF8000000000000
// CHECK: double 0x7FF4000000000000, double 0x7FF8000000000000

// The first line shows an unintended consequence.
// __builtin_nan() creates a legacy QNAN double with an empty payload
// (the first bit of the significand is clear to indicate quiet, so
// the second bit of the payload is set to maintain NAN-ness).
// The value is then truncated, but llvm::APFloat does not know about
// the inverted quiet bit, so it sets the first bit on conversion
// to indicate 'quiet' independently of the setting in clang.

float f[] = {
  __builtin_nan(""),
  __builtin_nans(""),
};

double d[] = {
  __builtin_nan(""),
  __builtin_nans(""),
};
