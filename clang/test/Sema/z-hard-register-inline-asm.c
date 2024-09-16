// RUN: %clang_cc1 %s -triple s390x-ibm-linux -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple s390x-ibm-zos -fsyntax-only -verify

void f1() {
  int a, b;
  register int c asm("r2");
  __asm("lhi %0,5\n"
        : "={r2}"(a)
        :);

  __asm("lgr %0,%1\n"
        : "={r2}"(a)
        : "{r1}"(b));

  __asm("lgr %0,%1\n"
        : "={r2}"(a)
        : "{%r1}"(b));

  __asm("lgr %0,%1\n"
        : "=&{r1}"(a)
        : "{r2}"(b));

  __asm("lhi %0,5\n"
        : "={r2"(a) // expected-error {{invalid output constraint '={r2' in asm}}
        :);

  __asm("lhi %0,5\n"
        : "={r17}"(a) // expected-error {{invalid output constraint '={r17}' in asm}}
        :);

  __asm("lhi %0,5\n"
        : "={}"(a) // expected-error {{invalid output constraint '={}' in asm}}
        :);

  __asm("lhi %0,5\n"
        : "=&{r2"(a) // expected-error {{invalid output constraint '=&{r2' in asm}}
        :);

  __asm("lgr %0,%1\n"
        : "=r"(a)
        : "{r1"(b)); // expected-error {{invalid input constraint '{r1' in asm}}

  __asm("lgr %0,%1\n"
        : "=r"(a)
        : "{}"(b)); // expected-error {{invalid input constraint '{}' in asm}}

  __asm("lgr %0,%1\n"
        : "={r1}"(a)
        : "{r17}"(b)); // expected-error {{invalid input constraint '{r17}' in asm}}
}
