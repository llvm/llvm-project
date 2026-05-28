// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <sanitizer/tysan_interface.h>

int main() {
  int i = 0;
  int *iPtr = &i;
  float *fPtr = (float *)iPtr;

  *fPtr = 5.0f;
  // CHECK: WRITE of size 4 at 0x{{.*}} with type float accesses an existing object of type int
  __tysan_set_type_unknown(iPtr, sizeof(int));
  *fPtr += 5.0f;
  // CHECK-NOT: WRITE of size 4 at 0x{{.*}} with type float accesses an existing object of type int

  *iPtr = 0;
  // CHECK: WRITE of size 4 at 0x{{.*}} with type int accesses an existing object of type float

  return 0;
}
