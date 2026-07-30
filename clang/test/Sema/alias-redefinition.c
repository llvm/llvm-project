// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -Wno-strict-prototypes -fsyntax-only -verify %s

void f0() {}
void fun0(void) __attribute((alias("f0")));

void f1() {}
void fun1() {} // expected-note {{previous definition}}
void fun1(void) __attribute((alias("f1"))); // expected-error {{redefinition of 'fun1'}}

void f2() {}
void fun2(void) __attribute((alias("f2"))); // expected-note {{previous definition}}
void fun2() {} // expected-error {{redefinition of 'fun2'}}

void f3() {}
void fun3(void) __attribute((alias("f3"))); // expected-note {{previous definition}}
void fun3(void) __attribute((alias("f3"))); // expected-error {{redefinition of 'fun3'}}

void f4() {}
void fun4(void) __attribute((alias("f4")));
void fun4(void);

void f5() {}
void __attribute((alias("f5"))) fun5(void) {} // expected-error {{definition 'fun5' cannot also be an alias}}

int var1 __attribute((alias("v1"))); // expected-error {{definition 'var1' cannot also be an alias}}
static int var2 __attribute((alias("v2"))) = 2; // expected-error {{definition 'var2' cannot also be an alias}}

extern int var3 __attribute__((alias("C"))); // expected-note{{previous definition is here}}
int var3 = 3; // expected-error{{redefinition of 'var3'}}

int var4; // expected-note{{previous definition is here}}
extern int var4 __attribute__((alias("v4"))); // expected-error{{alias definition of 'var4' after tentative definition}}
