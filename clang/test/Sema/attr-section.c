// RUN: %clang_cc1 -verify -fsyntax-only -triple x86_64-apple-darwin9 %s

int x __attribute__((section(
   42)));  // expected-error {{argument to section attribute was not a string literal}}


// rdar://4341926
int y __attribute__((section(
   "sadf"))); // expected-error {{mach-o section specifier requires a segment and section separated by a comma}}

// PR6007
void test() {
  __attribute__((section("NEAR,x"))) int n1; // expected-error {{'section' attribute is not valid on local variables}}
  __attribute__((section("NEAR,x"))) static int n2; // ok.
}

// pr9356
void __attribute__((section("foo,zed"))) test2(void); // expected-note {{previous attribute is here}}
void __attribute__((section("bar,zed"))) test2(void) {} // expected-warning {{section does not match previous declaration}}
