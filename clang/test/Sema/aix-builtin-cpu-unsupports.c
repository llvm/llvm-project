// RUN: %clang_cc1 -fsyntax-only -triple powerpc-ibm-aix7.2.0.0 -verify %s

int main(void) {
  if (__builtin_cpu_supports("aes")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("archpmu")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("dscr")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("ebb")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("htm-nosc")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("htm-no-suspend")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("ic_snoop")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("ieee128")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("isel")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("notb")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("scv")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("tar")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;

  if (__builtin_cpu_supports("vcrypto")) // expected-error {{invalid cpu feature string for builtin}}
    return 1;
}
