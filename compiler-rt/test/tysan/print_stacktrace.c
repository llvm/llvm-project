// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefixes=CHECK,CHECK-SHORT %s < %t.out

// RUN: %env_tysan_opts=print_stacktrace=1 %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefixes=CHECK,CHECK-LONG %s < %t.out

float *P;
void zero_array() {
  int i;
  for (i = 0; i < 1; ++i)
    P[i] = 0.0f;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 4 at {{.*}} with type float accesses an existing object of type p1 float
  // CHECK: {{#0 0x.* in zero_array .*print_stacktrace.c:}}[[@LINE-3]]
  // CHECK-SHORT-NOT: {{#1 0x.* in main .*print_stacktrace.c}}
  // CHECK-LONG-NEXT: {{#1 0x.* in main .*print_stacktrace.c}}
}

int main() {
  P = (float *)&P;
  zero_array();
}
