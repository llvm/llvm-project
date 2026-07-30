// RUN: %clang_cc1 %s -triple i686-apple-darwin -verify -fsyntax-only -fno-gnu-inline-asm

#if __has_extension(gnu_asm)
#error Expected extension 'gnu_asm' to be disabled
#endif

asm ("INST r1, 0"); // expected-error {{GNU-style inline assembly is disabled}}

void foo(void) __asm("__foo_func"); // AsmLabel is OK
int foo1 asm("bar1") = 0; // OK

asm(" "); // Whitespace is OK

void f (void) {
  long long foo = 0, bar;
  asm volatile("INST %0, %1" : "=r"(foo) : "r"(bar)); // expected-error {{GNU-style inline assembly is disabled}}
  asm (""); // Empty is OK
  return;
}
