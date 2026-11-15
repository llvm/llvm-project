// RUN: %clang_tysan -O0 %s -o %t && %run %t 10 >%t.out.0 2>&1
// RUN: FileCheck %s < %t.out.0
// RUN: %clang_tysan -O2 %s -o %t && %run %t 10 >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void __attribute__((noinline)) add_flt(float *a) {
  *a += 2.0f;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: READ of size 4 at {{.*}} with type float accesses an existing object of type int
  // CHECK: {{#0 0x.* in add_flt .*basic.c:}}[[@LINE-3]]
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 4 at {{.*}} with type float accesses an existing object of type int
  // CHECK: {{#0 0x.* in add_flt .*basic.c:}}[[@LINE-6]]
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: READ of size 4 at {{.*}} with type float accesses an existing object of type long
  // CHECK: {{#0 0x.* in add_flt .*basic.c:}}[[@LINE-9]]
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 4 at {{.*}} with type float accesses an existing object of type long
  // CHECK: {{#0 0x.* in add_flt .*basic.c:}}[[@LINE-12]]
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: READ of size 4 at {{.*}} with type float accesses part of an existing object of type long that starts at offset -4
  // CHECK: {{#0 0x.* in add_flt .*basic.c:}}[[@LINE-15]]
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 4 at {{.*}} with type float accesses part of an existing object of type long that starts at offset -4
  // CHECK: {{#0 0x.* in add_flt .*basic.c:}}[[@LINE-18]]
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: READ of size 4 at {{.*}} with type float partially accesses an object of type short that starts at offset 2
  // CHECK: {{#0 0x.* in add_flt .*basic.c:}}[[@LINE-21]]
}

int main(int argc, char *argv[]) {
  int x = atoi(argv[1]);
  add_flt((float *)&x);
  printf("x = %d\n", x);

  long y = x;
  add_flt((float *)&y);
  printf("y = %ld\n", y);

  add_flt(((float *)&y) + 1);
  printf("y = %ld\n", y);

  char *mem = (char *)malloc(4 * sizeof(short));
  memset(mem, 0, 4 * sizeof(short));
  *(short *)(mem + 2) = x;
  add_flt((float *)mem);
  short s1 = *(short *)mem;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: READ of size 2 at {{.*}} with type short accesses an existing object of type float
  // CHECK: {{#0 0x.* in main .*basic.c:}}[[@LINE-3]]
  short s2 = *(short *)(mem + 2);
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: READ of size 2 at {{.*}} with type short accesses part of an existing object of type float that starts at offset -2
  // CHECK: {{#0 0x.* in main .*basic.c:}}[[@LINE-3]]
  printf("m[0] = %d, m[1] = %d\n", s1, s2);
  free(mem);

  return 0;
}

// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation
