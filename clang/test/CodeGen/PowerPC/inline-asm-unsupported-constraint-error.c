// RUN: %clang_cc1 -emit-llvm -triple powerpc64le-unknown-unknown -verify %s
// This test case exist to test marking the 'a' inline assembly constraint as
// unsupported because powerpc previously marked it as supported.
int foo(int arg){
  asm goto ("bc %0,%1,%l[TEST_LABEL]" : : "a"(&&TEST_LABEL) : : TEST_LABEL); //expected-error {{invalid input constraint 'a' in asm}}
  return 0;
TEST_LABEL: return arg + 1;
}
