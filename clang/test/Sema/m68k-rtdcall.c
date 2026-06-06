// RUN: %clang_cc1 -triple m68k-unknown-unknown -mrtd -std=c89 -verify -verify=rtd %s
// RUN: %clang_cc1 -triple m68k-unknown-unknown -std=c89 -verify -verify=nortd %s

// rtd-error@+2 {{function with no prototype cannot use the m68k_rtd calling convention}}
void foo(int arg) {
  bar(arg);
}

// nortd-note@+4 {{previous declaration is here}}
// nortd-error@+4 {{function declared 'm68k_rtd' here was previously declared without calling convention}}
// nortd-note@+4 {{previous declaration is here}}
// nortd-error@+4 {{function declared 'm68k_rtd' here was previously declared without calling convention}}
void nonvariadic1(int a, int b, int c);
void __attribute__((m68k_rtd)) nonvariadic1(int a, int b, int c);
void nonvariadic2(int a, int b, int c);
void __attribute__((m68k_rtd)) nonvariadic2(int a, int b, int c) { }

// expected-error@+2 {{variadic function cannot use m68k_rtd calling convention}}
void variadic(int a, ...);
void __attribute__((m68k_rtd)) variadic(int a, ...);

// rtd-note@+2 {{previous declaration is here}}
// rtd-error@+2 {{redeclaration of 'a' with a different type: 'void ((*))(int, int) __attribute__((cdecl))' vs 'void (*)(int, int) __attribute__((m68k_rtd))'}}
extern void (*a)(int, int);
__attribute__((cdecl)) extern void (*a)(int, int);

extern void (*b)(int, ...);
__attribute__((cdecl)) extern void (*b)(int, ...);

// nortd-note@+2 {{previous declaration is here}}
// nortd-error@+2 {{redeclaration of 'c' with a different type: 'void ((*))(int, int) __attribute__((m68k_rtd))' vs 'void (*)(int, int)'}}
extern void (*c)(int, int);
__attribute__((m68k_rtd)) extern void (*c)(int, int);

// expected-error@+2 {{variadic function cannot use m68k_rtd calling convention}}
extern void (*d)(int, ...);
__attribute__((m68k_rtd)) extern void (*d)(int, ...);

// expected-warning@+1 {{'m68k_rtd' only applies to function types; type here is 'int'}}
__attribute__((m68k_rtd)) static int g = 0;

// expected-error@+1 {{'m68k_rtd' attribute takes no arguments}}
void __attribute__((m68k_rtd("invalid"))) z(int a);

// expected-error@+1 {{function with no prototype cannot use the m68k_rtd calling convention}}
void __attribute__((m68k_rtd)) e();
