// RUN: %clang_cc1 -triple x86_64-unknown-unknown -O1 -emit-llvm -o - %s | FileCheck %s

enum e0 { E0 };
struct s0 {
  enum e0         a:31;
};

int f0(void) {
  return __builtin_omp_required_simd_align(struct s0);
  // CHECK: ret i32 16
}
