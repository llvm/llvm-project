// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Note: In the final implementation, we will want these to generate
//       CIR-specific libc operations. This test is just a placeholder
//       to make sure we can compile these to normal function calls
//       until the special handling is implemented.

void *memcpy(void *, const void *, unsigned long);
void testMemcpy(void *dst, const void *src, unsigned long size) {
  memcpy(dst, src, size);
  // CHECK: cir.call @memcpy
}

void *memmove(void *, const void *, unsigned long);
void testMemmove(void *src, const void *dst, unsigned long size) {
  memmove(dst, src, size);
  // CHECK: cir.call @memmove
}

void *memset(void *, int, unsigned long);
void testMemset(void *dst, int val, unsigned long size) {
  memset(dst, val, size);
  // CHECK: cir.call @memset
}

double fabs(double);
double testFabs(double x) {
  return fabs(x);
  // CHECK: cir.call @fabs
}

float fabsf(float);
float testFabsf(float x) {
  return fabsf(x);
  // CHECK: cir.call @fabsf
}

int abs(int);
int testAbs(int x) {
  return abs(x);
  // CHECK: cir.call @abs
}

long labs(long);
long testLabs(long x) {
  return labs(x);
  // CHECK: cir.call @labs
}

long long llabs(long long);
long long testLlabs(long long x) {
  return llabs(x);
  // CHECK: cir.call @llabs
}
