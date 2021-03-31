// RUN: %clang %s -Xclang -verify -fsyntax-only
// RUN: %clang %s -cl-no-stdinc -Xclang -verify -DNOINC -fsyntax-only

#ifndef NOINC
//expected-no-diagnostics
#endif

void test() {
int i = get_global_id(0);
#ifdef NOINC
//expected-error@-2{{implicit declaration of function 'get_global_id' is invalid in OpenCL}}
#endif
}
