// RUN: %clang_cc1 %s -fsyntax-only

typedef void* image2d_t;

__kernel void f__ro(__read_only image2d_t a) { }

__kernel void f__wo(__write_only image2d_t a) { }

__kernel void f__rw(__read_write image2d_t a) { }


__kernel void fro(read_only image2d_t a) { }

__kernel void fwo(write_only image2d_t a) { }

__kernel void frw(read_write image2d_t a) { }
