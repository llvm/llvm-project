// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu  -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu  -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-win32  -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple i386-pc-win32  -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnux32  -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu  -fsyntax-only -verify %s -DNOCALLERSAVE=1

struct a {
  int b;
};

struct a test __attribute__((interrupt)); // expected-warning {{'interrupt' attribute only applies to non-K&R-style functions}}

__attribute__((interrupt)) int foo1(void) { return 0; }             // expected-error-re {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have a 'void' return type}}
__attribute__((interrupt)) void foo2(void) {}                       // expected-error-re {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have only a pointer parameter optionally followed by an integer parameter}}
__attribute__((interrupt)) void foo3(void *a, unsigned b, int c) {} // expected-error-re {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have only a pointer parameter optionally followed by an integer parameter}}
__attribute__((interrupt)) void foo4(int a) {}                      // expected-error-re {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have a pointer as the first parameter}}
#ifdef _LP64
// expected-error-re@+6 {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have a 'unsigned long' type as the second parameter}}
#elif defined(__x86_64__)
// expected-error-re@+4 {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have a 'unsigned long long' type as the second parameter}}
#else
// expected-error-re@+2 {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have a 'unsigned int' type as the second parameter}}
#endif
__attribute__((interrupt)) void foo5(void *a, float b) {}
#ifdef _LP64
// expected-error-re@+6 {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have a 'unsigned long' type as the second parameter}}
#elif defined(__x86_64__)
// expected-error-re@+4 {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have a 'unsigned long long' type as the second parameter}}
#else
// expected-error-re@+2 {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have a 'unsigned int' type as the second parameter}}
#endif
__attribute__((interrupt)) void foo6(float *a, int b) {}

#ifdef _LP64
// expected-error-re@+4 {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have a 'unsigned long' type as the second parameter}}
#elif defined(__x86_64__)
// expected-error-re@+2 {{{{(x86|x86-64)}} 'interrupt' attribute only applies to functions that have a 'unsigned long long' type as the second parameter}}
#endif
__attribute__((interrupt)) void foo7(int *a, unsigned b) {}
__attribute__((interrupt)) void foo8(int *a) {}

void g(void (*fp)(int *));
int main(int argc, char **argv) {
  void *ptr = (void *)&foo7;
  g(foo8);

  (void)ptr;
#ifndef __x86_64__
  // expected-error@+2 {{interrupt service routine cannot be called directly}}
#endif
  foo7((int *)argv, argc);
  foo8((int *)argv);       // expected-error {{interrupt service routine cannot be called directly}}
  return 0;
}

