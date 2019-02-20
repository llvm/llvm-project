// RUN: cp %s %t
// RUN: %clang_cc1 -cl-std=CL1.2 -pedantic -Wall -fixit %t
// RUN: %clang_cc1 -cl-std=CL1.2 -fsyntax-only -pedantic -Wall -Werror %t
// RUN: %clang_cc1 -cl-std=CL1.2 -E -o - %t | FileCheck %s

typedef __attribute__((ext_vector_type(4))) int int4;
typedef __attribute__((ext_vector_type(8))) int int8;

int printf(__constant const char* st, ...) __attribute__((format(printf, 1, 2)));


void vector_fixits() {
  printf("%v4f", (int4) 123);
  // CHECK: printf("%v4d", (int4) 123);

  printf("%v8d", (int4) 123);
  // CHECK: printf("%v4d", (int4) 123);

  printf("%v4d", (int8) 123);
  // CHECK: printf("%v8d", (int8) 123);

  printf("%v4f", (int8) 123);
  // CHECK: printf("%v8d", (int8) 123);
}
