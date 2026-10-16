// RUN: %clang_cc1 -fsyntax-only -triple powerpc-ibm-aix7.2.0.0 -verify %s

int main(void) {
  if (__builtin_cpu_supports("aes")) // expected-warning {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("archpmu")) // expected-warning {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("htm-nosc")) // expected-warning {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("htm-no-suspend")) // expected-warning {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("ic_snoop")) // expected-warning {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("ieee128")) // expected-warning {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("notb")) // expected-warning {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("scv")) // expected-warning {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("vcrypto")) // expected-warning {{invalid cpu feature string for builtin}}
    return 1;
}
