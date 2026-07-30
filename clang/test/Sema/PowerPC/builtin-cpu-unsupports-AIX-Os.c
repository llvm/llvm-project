// RUN: %clang_cc1 -fsyntax-only -triple  powerpc-ibm-aix7.1.0.0 -verify %s

int main(void) {
  if (__builtin_cpu_is("power8")) // expected-error {{this builtin is available only on AIX 7.2 and later operating systems}}
    return 1;

  if (__builtin_cpu_supports("power8")) // expected-error {{this builtin is available only on AIX 7.2 and later operating systems}}
    return 1;
}
