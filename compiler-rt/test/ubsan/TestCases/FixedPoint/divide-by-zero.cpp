// RUN: %clangxx -ffixed-point -fsanitize=fixed-point-divide-by-zero %s -o %t1 -DOP="0.0R / 0"
// RUN: %clangxx -ffixed-point -fsanitize=fixed-point-divide-by-zero %s -o %t2 -DOP="0.5R / 0"
// RUN: %clangxx -ffixed-point -fsanitize=fixed-point-divide-by-zero %s -o %t3 -DOP="0.0R / 0.0R"
// RUN: %clangxx -ffixed-point -fsanitize=fixed-point-divide-by-zero %s -o %t4 -DOP="0.5R / 0.0R"
// RUN: %clangxx -ffixed-point -fsanitize=fixed-point-divide-by-zero %s -o %t5 -DOP="_Accum a = 10.0K; a /= 0"

// RUN: %env_ubsan_opts=halt_on_error=1 not %run %t1 2>&1 | FileCheck %s
// RUN: %env_ubsan_opts=halt_on_error=1 not %run %t2 2>&1 | FileCheck %s
// RUN: %env_ubsan_opts=halt_on_error=1 not %run %t3 2>&1 | FileCheck %s
// RUN: %env_ubsan_opts=halt_on_error=1 not %run %t4 2>&1 | FileCheck %s
// RUN: %env_ubsan_opts=halt_on_error=1 not %run %t5 2>&1 | FileCheck %s

int main() {
  // CHECK: divide-by-zero.cpp:[[@LINE+1]]:3: runtime error: division by zero
  OP;
}
