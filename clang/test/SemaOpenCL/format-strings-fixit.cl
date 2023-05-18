// RUN: cp %s %t
// RUN: %clang_cc1 -cl-std=CL1.2 -pedantic -Wall -fixit %t -triple x86_64-unknown-linux-gnu
// RUN: %clang_cc1 -cl-std=CL1.2 -fsyntax-only -pedantic -Wall -Werror %t -triple x86_64-unknown-linux-gnu
// RUN: %clang_cc1 -cl-std=CL1.2 -E -o - %t -triple x86_64-unknown-linux-gnu | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef __attribute__((ext_vector_type(4))) char char4;
typedef __attribute__((ext_vector_type(4))) short short4;
typedef __attribute__((ext_vector_type(4))) int int4;
typedef __attribute__((ext_vector_type(4))) unsigned int uint4;
typedef __attribute__((ext_vector_type(8))) int int8;
typedef __attribute__((ext_vector_type(4))) long long4;
typedef __attribute__((ext_vector_type(4))) float float4;
typedef __attribute__((ext_vector_type(4))) double double4;

int printf(__constant const char* st, ...) __attribute__((format(printf, 1, 2)));


void vector_fixits() {
  printf("%v4f", (int4) 123);
  // CHECK: printf("%v4hld", (int4) 123);

  printf("%v8d", (int4) 123);
  // CHECK: printf("%v4hld", (int4) 123);

  printf("%v4d", (int8) 123);
  // CHECK: printf("%v8hld", (int8) 123);

  printf("%v4f", (int8) 123);
  // CHECK: printf("%v8hld", (int8) 123);

  printf("%v4ld", (int8) 123);
  // CHECK: printf("%v8hld", (int8) 123);

  printf("%v4hlf", (int4) 123);
  // CHECK: printf("%v4hld", (int4) 123);

  printf("%v8hld", (int4) 123);
  // CHECK: printf("%v4hld", (int4) 123);

  printf("%v4hld", (int8) 123);
  // CHECK: printf("%v8hld", (int8) 123);

  printf("%v4hlf", (int8) 123);
  // CHECK: printf("%v8hld", (int8) 123);

  printf("%v4hd", (int4) 123);
  // CHECK: printf("%v4hld", (int4) 123);

  printf("%v4hld", (short4) 123);
  // CHECK: printf("%v4hd", (short4) 123);

  printf("%v4ld", (short4) 123);
  // CHECK: printf("%v4hd", (short4) 123);

  printf("%v4hld", (long4) 123);
  // CHECK: printf("%v4ld", (long4) 123);

  printf("%v8f", (float4) 2.0f);
  // CHECK: printf("%v4hlf", (float4) 2.0f);

  printf("%v4f", (float4) 2.0f);
  // CHECK: printf("%v4hlf", (float4) 2.0f);

  printf("%v4lf", (double4) 2.0);
  // CHECK: printf("%v4lf", (double4) 2.0);

  /// FIXME: This should be fixed
  printf("%v4hhd", (int4) 123);
  // CHECK: printf("%v4hhd", (int4) 123);

  /// FIXME: This should be fixed
  printf("%v4hhd", (int8) 123);
  // CHECK: printf("%v4hhd", (int8) 123);
}
