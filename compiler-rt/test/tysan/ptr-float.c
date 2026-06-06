// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

float *P;
void zero_array() {
  int i;
  for (i = 0; i < 1; ++i)
    P[i] = 0.0f;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 4 at {{.*}} with type float accesses an existing object of type p1 float
  // CHECK: {{#0 0x.* in zero_array .*ptr-float.c:}}[[@LINE-3]]
}

int main() {
  P = (float *)&P;
  zero_array();
}

// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation
