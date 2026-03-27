// RUN: %clang_cc1 %s -triple s390x-ibm-linux -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple s390x-ibm-zos -fsyntax-only -verify

void f1() {
  int a, b;
  
  __asm("lgr %0,%1\n"
        : "={r2}"(a)
        : "{r1}{r3}"(b));
  
  __asm("lgr %0,%1\n"
        : "={r2}{r1}"(a)
        : "{r1}"(b));
  
  __asm("lgr %0,%1\n"
        : "={r2}"(a)
        : "{r1}{r3}r4}"(b)); // expected-error {{invalid input constraint '{r1}{r3}r4}' in asm}}
  
  __asm("lgr %0,%1\n"
        : "={r2}"(a)
        : "{r1r3}{r4}"(b)); // expected-error {{invalid input constraint '{r1r3}{r4}' in asm}}
  
  __asm("lgr %0,%1\n"
        : "={r1}{r3}r4}"(a) // expected-error {{invalid output constraint '={r1}{r3}r4}' in asm}}
        : "{r2}"(b));
  
  __asm("lgr %0,%1\n"
        : "={r1r3}{r4}"(a) // expected-error {{invalid output constraint '={r1r3}{r4}' in asm}}
        : "{r2}"(b));
}
