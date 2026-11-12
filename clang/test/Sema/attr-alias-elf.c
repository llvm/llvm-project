// RUN: %clang_cc1 -triple x86_64-pc-linux -Wno-strict-prototypes -verify -emit-llvm-only %s

void f1(void) __attribute__((alias("g1")));
void g1(void) {
}

void f2(void) __attribute__((alias("g2"))); // expected-error {{alias must point to a defined variable or function}}
// expected-note@-1 {{must refer to its mangled name}}

void f3(void) __attribute__((alias("g3"))); // expected-error {{alias must point to a defined variable or function}}
// expected-note@-1 {{must refer to its mangled name}}
void g3(void);


void f4() __attribute__((alias("g4")));
void g4() {}
void h4() __attribute__((alias("f4")));

void f5() __attribute__((alias("g5")));
void h5() __attribute__((alias("f5")));
void g5() {}

void g6() {}
void f6() __attribute__((alias("g6")));
void h6() __attribute__((alias("f6")));

void g7() {}
void h7() __attribute__((alias("f7")));
void f7() __attribute__((alias("g7")));

void h8() __attribute__((alias("f8")));
void g8() {}
void f8() __attribute__((alias("g8")));

void h9() __attribute__((alias("f9")));
void f9() __attribute__((alias("g9")));
void g9() {}

void f10() __attribute__((alias("g10"))); // expected-error {{alias definition is part of a cycle}}
void g10() __attribute__((alias("f10"))); // expected-error {{alias definition is part of a cycle}}

// FIXME: This could be a bit better, h10 is not part of the cycle, it points
// to it.
void h10() __attribute__((alias("g10"))); // expected-error {{alias definition is part of a cycle}}

extern int a1 __attribute__((alias("b1")));
int b1 = 42;

extern int a2 __attribute__((alias("b2"))); // expected-error {{alias must point to a defined variable or function}}
// expected-note@-1 {{must refer to its mangled name}}

extern int a3 __attribute__((alias("b3"))); // expected-error {{alias must point to a defined variable or function}}
// expected-note@-1 {{must refer to its mangled name}}
extern int b3;

extern int a4 __attribute__((alias("b4"))); // expected-error {{alias must point to a defined variable or function}}
// expected-note@-1 {{must refer to its mangled name}}
typedef int b4;

void test2_bar() {}
void test2_foo() __attribute__((weak, alias("test2_bar")));
void test2_zed() __attribute__((alias("test2_foo"))); // expected-warning {{alias will always resolve to test2_bar even if weak definition of test2_foo is overridden}}

void test3_bar() { }
void test3_foo() __attribute__((section("test"))); // expected-warning {{alias will not be in section 'test' but in the same section as the aliasee}}
void test3_foo() __attribute__((alias("test3_bar")));

__attribute__((section("test"))) void test4_bar() { }
void test4_foo() __attribute__((section("test")));
void test4_foo() __attribute__((alias("test4_bar")));

int test5_bar = 0;
extern struct incomplete_type test5_foo __attribute__((alias("test5_bar")));

int test6 = 0;
// expected-note@-1 {{aliasee is declared here}}
void test6_alias() __attribute__((alias("test6")));
// expected-error@-1 {{cannot alias a variable with a function}}

extern int test7_alias __attribute__((alias("test7")));
// expected-error@-1 {{cannot alias a function with a variable}}
int test7(int x) { return x * 2; }
// expected-note@-1 {{aliasee is declared here}}

void *test8_ifunc() { return 0; }
void test8(void) __attribute__((ifunc("test8_ifunc")));
// expected-note@-1 {{aliasee is declared here}}
extern int test8_alias __attribute__((alias("test8")));
// expected-error@-1 {{cannot alias a function with a variable}}

void test9() {}
// expected-note@-1 {{aliasee is declared here}}
int test9_alias() __attribute__((alias("test9")));
// expected-warning@-1 {{alias and aliasee have different types 'int ()' and 'void ()'}}

// No warning for an alias with unspecified parameters if the return types match.
int test10(int x, int y) { return x + y; }
int test10_alias() __attribute__((alias("test10")));

// No warning for an alias target with unspecified parameters if the return types match.
int test11() { return 7; }
int test11_alias(int x) __attribute__((alias("test11")));

int test12(int x, int y) { return x + y; }
// expected-note@-1 {{aliasee is declared here}}
int test12_alias(int x, ...) __attribute__((alias("test12")));
// expected-warning@-1 {{alias and aliasee have different types 'int (int, ...)' and 'int (int, int)'}}

// No warning when using typedef equivalents.
typedef int Integer;
Integer test13(int x) { return x; }
int test13_alias(Integer) __attribute__((alias("test13")));

// Compiler-generated variables are not valid alias targets.
char *test14 = "asdf";
extern char test14_alias[5] __attribute__((alias(".str")));
// expected-error@-1 {{alias must point to a defined variable or function}}

// Unprototyped functions should not alias variadic function and vice versa.
int test15() { return 9; }
// expected-note@-1 {{aliasee is declared here}}
int test15_alias(int x, ...) __attribute__((alias("test15")));
// expected-warning@-1 {{alias and aliasee have different types 'int (int, ...)' and 'int ()'}}

void test16(int x, ...) { }
// expected-note@-1 {{aliasee is declared here}}
void test16_alias() __attribute__((alias("test16")));
// expected-warning@-1 {{alias and aliasee have different types 'void ()' and 'void (int, ...)'}}
