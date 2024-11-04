// Test this without pch.
// RUN: %clang_cc1 -fblocks -include %S/blocks.h -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -fblocks -o %t %S/blocks.h
// RUN: %clang_cc1 -fblocks -include-pch %t -emit-llvm -o - %s 

int do_add(int x, int y) { return add(x, y); }

int do_scaled_add(int a, int b, int s) {
  return scaled_add(a, b, s);
}
