// Test if PGO sample use preparation passes are executed correctly.
//
// Ensure that instcombine is executed after simplifycfg and sroa so that
// "a < 255" will not be converted to a * 256 < 255 * 256.
// RUN: %clang_cc1 -O2 -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s -emit-llvm -o - 2>&1 | FileCheck %s

void bar(int);
void foo(int x, int y, int z) {
  int m;
  for (m = 0; m < x ; m++) {
    int a = (((y >> 8) & 0xff) * z) / 256;
    bar(a < 255 ? a : 255);
  }
}

// CHECK-NOT: icmp slt i32 %mul, 65280
