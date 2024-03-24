// RUN: %clang_cc1 -S -emit-llvm -o - %s

typedef __attribute__((__ext_vector_type__(4))) _Bool BoolVector;

BoolVector vec;

void f(int i, int j) {
  vec[i] |= vec[j];
}