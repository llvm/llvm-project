// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1 && FileCheck --check-prefix=CHECK-SANITIZED %s < %t.out
// RUN: %clang_tysan -DNOSAN -O0 %s -o %t && %run %t >%t.out 2>&1 && FileCheck --check-prefix=CHECK-NOSAN %s < %t.out
// RUN: %clang -O0 %s -o %t && %run %t >%t.out 2>&1 && FileCheck --check-prefix=CHECK-SIMPLE %s < %t.out

#include <stdio.h>

#if __has_feature(type_sanitizer)

#  ifdef NOSAN
__attribute__((no_sanitize("type")))
#  endif
int main(){

  int value = 42;
  printf("As float: %f\n", *(float *)&value);
  // CHECK-SANITIZED: ERROR: TypeSanitizer
  // CHECK-NOSAN-NOT: ERROR: TypeSanitizer

  return 0;
}

#else

int main() {
  printf("Nothing interesting here\n");
  return 0;
}
// CHECK-SIMPLE: Nothing interesting here

#endif
