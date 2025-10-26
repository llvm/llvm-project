// RUN: %clangxx -DOP="0.0R / 0" -ffixed-point -fsanitize=fixed-point-divide-by-zero %s -o %t1 && %run %t1 2>&1 | FileCheck %s
// RUN: %clangxx -DOP="0.5R / 0" -ffixed-point -fsanitize=fixed-point-divide-by-zero %s -o %t2 && %run %t2 2>&1 | FileCheck %s
// RUN: %clangxx -DOP="0.0R / 0.0R" -ffixed-point -fsanitize=fixed-point-divide-by-zero %s -o %t3 && %run %t3 2>&1 | FileCheck %s
// RUN: %clangxx -DOP="0.5R / 0.0R" -ffixed-point -fsanitize=fixed-point-divide-by-zero %s -o %t4 && %run %t4 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=undefined -DOP="_Fract a = 0.5R; a /= 0" -ffixed-point -fsanitize=fixed-point-divide-by-zero %s -o %t5 && %run %t5 2>&1 | FileCheck %s

int main() {
  // CHECK: divide-by-zero.cpp:[[@LINE+1]]:3: runtime error: division by zero
  OP;
}
