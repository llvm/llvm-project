// RUN: %clang_cc1 -triple x86_64-windows -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-linux -fsyntax-only -verify -emit-llvm-only -DCHECK_ALIASES %s
// RUN: %clang_cc1 -triple x86_64-linux -fsyntax-only -verify -emit-llvm-only %s

#if defined(_WIN32)
void foo() {}
void bar() __attribute__((ifunc("foo")));
//expected-warning@-1 {{unknown attribute 'ifunc' ignored}}

#else
#if defined(CHECK_ALIASES)
void* f1_ifunc();
void f1() __attribute__((ifunc("f1_ifunc")));
//expected-error@-1 {{ifunc must point to a defined function}}

void* f2_a() __attribute__((ifunc("f2_b")));
//expected-error@-1 {{ifunc definition is part of a cycle}}
void* f2_b() __attribute__((ifunc("f2_a")));
//expected-error@-1 {{ifunc definition is part of a cycle}}

void* f3_a() __attribute__((ifunc("f3_b")));
//expected-warning@-1 {{ifunc will always resolve to f3_c even if weak definition of f3_b is overridden}}
void* f3_b() __attribute__((weak, alias("f3_c")));
void* f3_c() { return 0; }

void f4_ifunc() {}
void f4() __attribute__((ifunc("f4_ifunc")));
//expected-error@-1 {{ifunc resolver function must return a pointer}}

#else
void f1a() __asm("f1");
void f1a() {}
//expected-note@-1 {{previous definition is here}}
void f1() __attribute__((ifunc("f1_ifunc")));
//expected-error@-1 {{definition with same mangled name 'f1' as another definition}}
void* f1_ifunc() { return 0; }

void* f6_ifunc(int i);
void __attribute__((ifunc("f6_ifunc"))) f6() {}
//expected-error@-1 {{definition 'f6' cannot also be an ifunc}}

#endif
#endif
