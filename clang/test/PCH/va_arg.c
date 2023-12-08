// Test this without pch.
// RUN: %clang_cc1 -triple=x86_64-unknown-freebsd7.0 -include %S/va_arg.h %s -emit-llvm -o -
// REQUIRES: x86-registered-target

// Test with pch.
// RUN: %clang_cc1 -triple=x86_64-unknown-freebsd7.0 -emit-pch -o %t %S/va_arg.h
// RUN: %clang_cc1 -triple=x86_64-unknown-freebsd7.0 -include-pch %t %s -emit-llvm -o -

char *g0(char** argv, int argc) { return argv[argc]; }

char *g(char **argv) {
  f(g0, argv, 1, 2, 3);
}

char *i0(char **argv, int argc) { return argv[argc]; }

char *i(char **argv) {
  h(i0, argv, 1, 2, 3);
}
