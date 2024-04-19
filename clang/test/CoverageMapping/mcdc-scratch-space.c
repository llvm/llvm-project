// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c99 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s
// XFAIL: *
// REQUIRES: asserts

int builtin_macro0(int a) {
  return (__LINE__
          && a);
}

int builtin_macro1(int a) {
  return (a
          || __LINE__);
}

#define PRE(x) pre_##x

int pre0(int pre_a, int b_post) {
  return (PRE(a)
          && b_post);
}

#define POST(x) x##_post

int post0(int pre_a, int b_post) {
  return (pre_a
          || POST(b));
}
