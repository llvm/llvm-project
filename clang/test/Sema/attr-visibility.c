// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify %s

void test0() __attribute__((visibility("default")));
void test1() __attribute__((visibility("hidden")));
void test2() __attribute__((visibility("internal")));

// rdar://problem/10753392
void test3() __attribute__((visibility("protected"))); // expected-warning {{target does not support 'protected' visibility; using 'default'}}

struct __attribute__((visibility("hidden"))) test4; // expected-note {{previous attribute is here}}
struct test4;
struct __attribute__((visibility("default"))) test4; // expected-error {{visibility does not match previous declaration}}

struct test5;
struct __attribute__((visibility("hidden"))) test5; // expected-note {{previous attribute is here}}
struct __attribute__((visibility("default"))) test5; // expected-error {{visibility does not match previous declaration}}

void test6() __attribute__((visibility("hidden"), // expected-note {{previous attribute is here}}
                            visibility("default"))); // expected-error {{visibility does not match previous declaration}}

extern int test7 __attribute__((visibility("default"))); // expected-note {{previous attribute is here}}
extern int test7 __attribute__((visibility("hidden"))); // expected-error {{visibility does not match previous declaration}}

typedef int __attribute__((visibility("default"))) bar; // expected-warning {{visibility attribute ignored}}
