// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify %s

#pragma alloc_text()        // expected-warning {{expected a string literal for the section name in '#pragma alloc_text'}}
#pragma alloc_text(a        // expected-warning {{expected ',' in '#pragma alloc_text'}}
#pragma alloc_text(a, a     // expected-warning {{missing ')' after '#pragma alloc_text'}}
#pragma alloc_text(a, a)    // expected-error {{use of undeclared a}}
#pragma alloc_text(L"a", a) // expected-warning {{expected a string literal for the section name}}

void foo();
#pragma alloc_text(a, foo) // expected-error {{'#pragma alloc_text' is applicable only to functions with C linkage}}

extern "C" void foo1();
#pragma alloc_text(a, foo1)      // no-warning
#pragma alloc_text(a, foo1) asdf // expected-warning {{extra tokens at end of '#pragma alloc_text'}}
#pragma alloc_text(a, foo1       // expected-warning {{missing ')' after '#pragma alloc_text'}}

namespace N {
#pragma alloc_text(b, foo1) // no-warning
}

extern "C" {
void foo2();
#pragma alloc_text(a, foo2) // no-warning
}

void foo3() {
#pragma alloc_text(a, foo1) // expected-error {{'#pragma alloc_text' can only appear at file scope}}
}

extern "C" void foo4();
#pragma alloc_text(c, foo4) // no-warning
void foo4() {}

void foo5();                // expected-note {{previous declaration is here}}
#pragma alloc_text(c, foo5) // expected-error {{'#pragma alloc_text' is applicable only to functions with C linkage}}
extern "C" void foo5() {}   // expected-error {{declaration of 'foo5' has a different language linkage}}

extern "C" {
static void foo6();
#pragma alloc_text(c, foo6) // no-warning
void foo6() {}
}

extern "C" {
static void foo7();
}
static void foo7();
#pragma alloc_text(c, foo7) // no-warning
void foo7() {}

static void foo8();
extern "C" {
static void foo8();
}
#pragma alloc_text(c, foo8) // expected-error {{'#pragma alloc_text' is applicable only to functions with C linkage}}
void foo8() {}

enum foo9 { A, B, C };
#pragma alloc_text(c, foo9) // expected-error {{'#pragma alloc_text' is applicable only to functions}}
