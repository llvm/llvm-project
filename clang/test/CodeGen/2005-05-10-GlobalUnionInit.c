// RUN: %clang_cc1 %s -emit-llvm -o -

union A {                    // { uint }
  union B { double *C; } D;
} E = { { (double*)12312 } };

