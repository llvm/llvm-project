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

typedef void (*func_ptr)(void);

static void implementation(void) {}

static func_ptr resolver1(void) {
  return implementation;
}

void f6(void) __attribute__((ifunc("resolver1"))); // expected-note {{previous definition is here}}
void f6(void) __attribute__((alias("implementation"))); // expected-error {{redefinition of 'f6'}}

void f7(void) __attribute__((alias("implementation"))); // expected-note {{previous definition is here}}
void f7(void) __attribute__((ifunc("resolver1"))); // expected-error {{redefinition of 'f7'}}

void f8(void) __attribute__((ifunc("resolver1"), alias("implementation"))); // expected-error {{definition 'f8' cannot also be an alias}}

void f9(void) __attribute__((alias("implementation"), ifunc("resolver1"))); // expected-error {{definition 'f9' cannot also be an ifunc}}

int var1 __attribute((alias("v1"))); // expected-error {{definition 'var1' cannot also be an alias}}
static int var2 __attribute((alias("v2"))) = 2; // expected-error {{definition 'var2' cannot also be an alias}}
extern int var_with_extern_initializer __attribute__((alias(""))) = 42; // expected-error {{definition 'var_with_extern_initializer' cannot also be an alias}}
// expected-warning@-1 {{'extern' variable has an initializer}}
extern int var_with_extern_initializer1 __attribute__((alias("v1"))) = 42; // expected-error {{definition 'var_with_extern_initializer1' cannot also be an alias}}
// expected-warning@-1 {{'extern' variable has an initializer}}

int target;
int loader_then_alias __attribute((loader_uninitialized, alias("target"))); // expected-error {{definition 'loader_then_alias' cannot also be an alias}}

int alias_then_loader __attribute((alias("target"), loader_uninitialized)); // expected-error {{definition 'alias_then_loader' cannot also be an alias}}

int loader_redecl_alias __attribute((loader_uninitialized)); // expected-note {{previous definition is here}}
extern int loader_redecl_alias __attribute((alias("target"))); // expected-error {{redefinition of 'loader_redecl_alias'}}

extern int loader_redecl_alias1 __attribute((alias("target"))); // expected-note {{previous definition is here}}
int loader_redecl_alias1 __attribute((loader_uninitialized)); // expected-error {{redeclaration cannot add 'loader_uninitialized' attribute}}

extern int var3 __attribute__((alias("C"))); // expected-note{{previous definition is here}}
int var3 = 3; // expected-error{{redefinition of 'var3'}}

int var4; // expected-note{{previous definition is here}}
extern int var4 __attribute__((alias("v4"))); // expected-error{{alias definition of 'var4' after tentative definition}}
