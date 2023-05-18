// RUN: %clang_cc1 -triple nvptx-unknown-unknown -O3 -S -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -O3 -S -o %t %s -emit-llvm

// Just make sure Clang uses the proper calling convention for the NVPTX back-end.
// If something is wrong, the back-end will fail.
void foo(float* a,
         float* b) {
  a[0] = b[0];
}
