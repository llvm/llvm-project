// RUN: %clang_cc1 -emit-llvm-only -triple powerpc64-ibm-aix-xcoff -verify %s
// RUN: %clang_cc1 -emit-llvm-only -triple powerpc-ibm-aix-xcoff -verify %s
// This test file validates that we diagnose inline assembly constraint error
// in the clang front-end as expected.

 int labelConstraintError(int arg){
  asm goto ("bc 12,2,%l[TEST_LABEL]" : : "s"(&&TEST_LABEL) : : TEST_LABEL); //expected-error {{invalid input constraint 's' in asm}}
  return 0;
TEST_LABEL: return arg + 1;
}

char wrongAddrConstraint(char* result) {
  asm ("stb %1,%0" : "a"(result) : "r"('E') :); //expected-error {{invalid output constraint 'a' in asm}}
  return *result;
}
