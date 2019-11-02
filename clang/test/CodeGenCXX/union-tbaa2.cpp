// RUN: %clang_cc1 %s -O2 -fno-experimental-new-pass-manager -std=c++11 -triple x86_64-unknown-linux-gnu -target-cpu x86-64 -target-feature +sse4.2 -target-feature +avx -emit-llvm -o - | FileCheck %s

// Testcase from llvm.org/PR32056

extern "C" int printf (const char *__restrict __format, ...);

typedef double __m256d __attribute__((__vector_size__(32)));

static __inline __m256d __attribute__((__always_inline__, __nodebug__,
                                       __target__("avx")))
_mm256_setr_pd(double __a, double __b, double __c, double __d) {
  return (__m256d){ __a, __b, __c, __d };
}

struct A {
  A () {
// Check that the TBAA information generated for the stores to the
// union members is based on the omnipotent char.
// CHECK: store <4 x double>
// CHECK: tbaa ![[OCPATH:[0-9]+]]
// CHECK: store <4 x double>
// CHECK: tbaa ![[OCPATH]]
// CHECK: call
    a = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
    b = _mm256_setr_pd(4.0, 5.0, 6.0, 7.0);
  }

  const double *begin() { return c; }
  const double *end() { return c+8; }

  union {
    struct { __m256d a, b; };
    double c[8];
  };
};

int main(int argc, char *argv[]) {
  A a;
  for (double value : a)
    printf("%f ", value);
  return 0;
}

// CHECK-DAG: ![[CHAR:[0-9]+]] = !{!"omnipotent char"
// CHECK-DAG: ![[OCPATH]] = !{![[CHAR]], ![[CHAR]], i64 0}
