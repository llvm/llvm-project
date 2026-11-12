// RUN: %clang_cc1 %s -O3 -emit-llvm -o - | grep mul
// PR1233

float foo(int w, float A[][w], int g, int h) {
  return A[g][0];
}

