// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -fsyntax-only

// expected-no-diagnostics

// Check redefinition of standard types
typedef atomic_int atomic_flag;
