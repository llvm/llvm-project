// RUN: %clang_cc1 %s -fsyntax-only -verify
// RUN: %clang_cc1 %s -fsyntax-only -verify -cl-std=CL2.0 -DCL20
// expected-no-diagnostics

__kernel void f__ro(__read_only image2d_t a) { }

__kernel void f__wo(__write_only image2d_t a) { }

#if CL20
__kernel void f__rw(__read_write image2d_t a) { }
#endif

__kernel void fro(read_only image2d_t a) { }

__kernel void fwo(write_only image2d_t a) { }

#if CL20
__kernel void frw(read_write image2d_t a) { }
#endif
