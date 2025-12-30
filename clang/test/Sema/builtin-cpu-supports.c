// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-linux-gnu -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple aarch64-linux-gnu -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple riscv32-linux-gnu -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple riscv64-linux-gnu -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple powerpc64le-unknown-linux -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple powerpc64-unknown-aix7.2.0.0 -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple powerpc-unknown-aix7.2.0.0 -verify %s

extern void a(const char *);

extern const char *str;

int main(void) {
#ifdef __x86_64__
  if (__builtin_cpu_supports("ss")) // expected-warning {{invalid cpu feature string}}
    a("sse4.2");

  if (__builtin_cpu_supports(str)) // expected-error {{expression is not a string literal}}
    a(str);

  if (__builtin_cpu_is("int")) // expected-error {{invalid cpu name for builtin}}
    a("intel");

  (void)__builtin_cpu_is("x86-64");    // expected-error {{invalid cpu name for builtin}}
  (void)__builtin_cpu_is("x86-64-v2"); // expected-error {{invalid cpu name for builtin}}
  (void)__builtin_cpu_is("x86-64-v3"); // expected-error {{invalid cpu name for builtin}}
  (void)__builtin_cpu_is("x86-64-v4"); // expected-error {{invalid cpu name for builtin}}

  (void)__builtin_cpu_supports("x86-64");
  (void)__builtin_cpu_supports("x86-64-v2");
  (void)__builtin_cpu_supports("x86-64-v3");
  (void)__builtin_cpu_supports("x86-64-v4");
  (void)__builtin_cpu_supports("x86-64-v5"); // expected-warning {{invalid cpu feature string for builtin}}
#endif

#ifdef __aarch64__
  if (__builtin_cpu_supports("neon")) // expected-warning {{invalid cpu feature string for builtin}}
    a("vsx");

  if (__builtin_cpu_is("cortex-x3")) // expected-error {{builtin is not supported on this target}}
    a("pwr9");

  __builtin_cpu_init(); // expected-error {{builtin is not supported on this target}}
#endif

#ifdef __riscv
  if (__builtin_cpu_supports("garbage")) // expected-warning {{invalid cpu feature string for builtin}}
    a("vsx");
#endif

#ifdef __powerpc__
  if (__builtin_cpu_is("garbage")) // expected-error {{invalid cpu name for builtin}}
    a("vsx");

  if (__builtin_cpu_is("power3")) // expected-error {{invalid cpu name for builtin}}
    a("vsx");

  if (__builtin_cpu_supports("garbage")) // expected-warning {{invalid cpu feature string for builtin}}
    a("vsx");
#endif

  return 0;
}
