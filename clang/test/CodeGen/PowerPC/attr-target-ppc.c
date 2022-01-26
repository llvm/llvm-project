// RUN: not %clang_cc1 -triple powerpc64le-linux-gnu -emit-llvm %s -o -

long __attribute__((target("power8-vector,no-vsx"))) foo (void) { return 0; }  // expected-error {{option '-mpower8-vector' cannot be specified with '-mno-vsx'}}

