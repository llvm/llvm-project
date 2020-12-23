// RUN: %clang_cc1 -verify -fsyntax-only -triple x86_64-apple-darwin9 %s

int x __attribute__((section(
   42)));  // expected-error {{'section' attribute requires a string}}


// rdar://4341926
int y __attribute__((section(
   "sadf"))); // expected-error {{mach-o section specifier requires a segment and section separated by a comma}}

// PR6007
void test() {
  __attribute__((section("NEAR,x"))) int n1; // expected-error {{'section' attribute only applies to functions, global variables, Objective-C methods, and Objective-C properties}}
  __attribute__((section("NEAR,x"))) static int n2; // ok.
}

// pr9356
void __attribute__((section("foo,zed"))) test2(void); // expected-note {{previous attribute is here}}
void __attribute__((section("bar,zed"))) test2(void) {} // expected-warning {{section does not match previous declaration}}

enum __attribute__((section("NEAR,x"))) e { one }; // expected-error {{'section' attribute only applies to}}

extern int a; // expected-note {{previous declaration is here}}
int *b = &a;
extern int a __attribute__((section("foo,zed"))); // expected-warning {{section attribute is specified on redeclared variable}}

// Not a warning.
int c;
int c __attribute__((section("seg1,sec1")));

// Also OK.
struct r_debug {};
extern struct r_debug _r_debug;
struct r_debug _r_debug __attribute__((nocommon, section(".r_debug,bar")));

// Section type conflicts between functions and variables
void test3(void) __attribute__((section("seg3,sec3"))); // expected-note {{declared here}}
void test3(void) {}
const int const_global_var __attribute__((section("seg3,sec3"))) = 10; // expected-error {{'const_global_var' causes a section type conflict with 'test3'}}

void test4(void) __attribute__((section("seg4,sec4"))); // expected-note {{declared here}}
void test4(void) {}
int mut_global_var __attribute__((section("seg4,sec4"))) = 10; // expected-error {{'mut_global_var' causes a section type conflict with 'test4'}}

const int global_seg5sec5 __attribute__((section("seg5,sec5"))) = 10; // expected-note {{declared here}}
void test5(void) __attribute__((section("seg5,sec5")));               // expected-error {{'test5' causes a section type conflict with 'global_seg5sec5'}}
void test5(void) {}

void test6(void);
const int global_seg6sec6 __attribute__((section("seg6,sec6"))) = 10; // expected-note {{declared here}}
void test6(void) __attribute__((section("seg6,sec6")));               // expected-error {{'test6' causes a section type conflict with 'global_seg6sec6'}}
void test6(void) {}
