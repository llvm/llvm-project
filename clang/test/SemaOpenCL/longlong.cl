// RUN: %clang_cc1 %s -cl-std=CL1.0 -verify -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=CL1.2 -verify -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=CL3.0 -verify -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=CLC++ -verify -fsyntax-only

void kernel test_longlong() {
  long long x = 0;          // expected-warning{{'long long' is a reserved data type in OpenCL C}}
  unsigned long long y = 0; // expected-warning{{'long long' is a reserved data type in OpenCL C}}
  typedef long long longlong2 __attribute__((ext_vector_type(2))); // expected-warning{{'long long' is a reserved data type in OpenCL C}}
  typedef unsigned long long ulonglong2 __attribute__((ext_vector_type(4))); // expected-warning{{'long long' is a reserved data type in OpenCL C}}
}
