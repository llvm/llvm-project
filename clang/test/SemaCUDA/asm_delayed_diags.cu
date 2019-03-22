// RUN: %clang_cc1 -fsyntax-only -verify %s -DHOST -triple x86_64-unknown-linux-gnu -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -verify %s -DHOST -DHOST_USED -triple x86_64-unknown-linux-gnu -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify %s -DDEVICE_NOT_USED -triple nvptx-unknown-cuda -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify %s -DDEVICE -triple nvptx-unknown-cuda -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify %s -DDEVICE -DDEVICE_USED -triple nvptx-unknown-cuda -Wuninitialized

// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

#if (defined(HOST) && !defined(HOST_USED)) || defined(DEVICE_NOT_USED)
// expected-no-diagnostics
#endif

#include "Inputs/cuda.h"

static __device__ __host__ void t1(int r) {
  __asm__("PR3908 %[lf] %[xx] %[li] %[r]"
          : [ r ] "+r"(r)
          : [ lf ] "mx"(0), [ li ] "mr"(0), [ xx ] "x"((double)(0)));
}

static __device__ __host__ unsigned t2(signed char input) {
  unsigned output;
  __asm__("xyz"
          : "=a"(output)
          : "0"(input));
  return output;
}

static __device__ __host__ double t3(double x) {
  register long double result;
  __asm __volatile("frndint"
                   : "=t"(result)
                   : "0"(x));
  return result;
}

static __device__ __host__ unsigned char t4(unsigned char a, unsigned char b) {
  unsigned int la = a;
  unsigned int lb = b;
  unsigned int bigres;
  unsigned char res;
  __asm__("0:\n1:\n"
          : [ bigres ] "=la"(bigres)
          : [ la ] "0"(la), [ lb ] "c"(lb)
          : "edx", "cc");
  res = bigres;
  return res;
}

static __device__ __host__ void t5(void) {
  __asm__ __volatile__(
      "finit"
      :
      :
      : "st", "st(1)", "st(2)", "st(3)",
        "st(4)", "st(5)", "st(6)", "st(7)",
        "fpsr", "fpcr");
}

typedef long long __m256i __attribute__((__vector_size__(32)));
static __device__ __host__ void t6(__m256i *p) {
  __asm__ volatile("vmovaps  %0, %%ymm0" ::"m"(*(__m256i *)p)
                   : "ymm0");
}

static __device__ __host__ void t7(__m256i *p) {
  __asm__ volatile("vmovaps  %0, %%ymm0" ::"m"(*(__m256i *)p)
                   : "r0");
}

#ifdef DEVICE
__device__ int m() {
  t1(0);
  t2(0);
  t3(0);
  t4(0, 0);
  t5();
  t6(0);
#ifdef DEVICE_USED
  t7(0);
#endif // DEVICE_USED
  return 0;
}
#endif // DEVICE

#ifdef HOST
__host__ int main() {
  t1(0);
  t2(0);
  t3(0);
  t4(0, 0);
  t5();
  t6(0);
#ifdef HOST_USED
  t7(0);
#endif // HOST_USED
  return 0;
}
#endif // HOST

#if defined(HOST_USED)
// expected-error@69 {{unknown register name 'r0' in asm}}
// expected-note@96 {{called by 'main'}}
#elif defined(DEVICE)
// expected-error@19 {{invalid input constraint 'mx' in asm}}
// expected-error@25 {{invalid output constraint '=a' in asm}}
// expected-error@33 {{invalid output constraint '=t' in asm}}
// expected-error@44 {{invalid output constraint '=la' in asm}}
// expected-error@56 {{unknown register name 'st' in asm}}
// expected-error@64 {{unknown register name 'ymm0' in asm}}
// expected-note@74 {{called by 'm'}}
// expected-note@75 {{called by 'm'}}
// expected-note@76 {{called by 'm'}}
// expected-note@77 {{called by 'm'}}
// expected-note@78 {{called by 'm'}}
// expected-note@79 {{called by 'm'}}
#endif
