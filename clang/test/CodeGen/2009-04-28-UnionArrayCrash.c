// RUN: %clang_cc1 -emit-llvm %s -o - 
// PR4082
union U {
  int I;
  double F;
};

union U arr[] = { { .I = 4 }, { .F = 123.} };
union U *P = &arr[0];


