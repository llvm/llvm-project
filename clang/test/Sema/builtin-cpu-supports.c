// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-linux-gnu -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple powerpc64le-linux-gnu -verify %s

extern void a(const char *);

extern const char *str;

int main(void) {
#ifdef __x86_64__
  if (__builtin_cpu_supports("ss")) // expected-error {{invalid cpu feature string}}
    a("sse4.2");

  if (__builtin_cpu_supports(str)) // expected-error {{expression is not a string literal}}
    a(str);

  if (__builtin_cpu_is("int")) // expected-error {{invalid cpu name for builtin}}
    a("intel");

  (void)__builtin_cpu_is("x86-64");    // expected-error {{invalid cpu name for builtin}}
  (void)__builtin_cpu_is("x86-64-v2"); // expected-error {{invalid cpu name for builtin}}
  (void)__builtin_cpu_is("x86-64-v3"); // expected-error {{invalid cpu name for builtin}}
  (void)__builtin_cpu_is("x86-64-v4"); // expected-error {{invalid cpu name for builtin}}
#else
  if (__builtin_cpu_supports("vsx")) // expected-error {{use of unknown builtin}}
    a("vsx");

  if (__builtin_cpu_is("pwr9")) // expected-error {{use of unknown builtin}}
    a("pwr9");
#endif

  return 0;
}
