// RUN: %clang_cc1  -fsyntax-only -verify -Wno-unused-value %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wcast-of-sel-type -Wno-unused-value %s
// rdar://12107381

SEL s;

SEL sel_registerName(const char *);

int main() {
(char *)s;  // expected-warning {{cast of type 'SEL' to 'char *' is deprecated; use sel_getName instead}}
(void *)s;  // ok
(const char *)sel_registerName("foo");  // expected-warning {{cast of type 'SEL' to 'const char *' is deprecated; use sel_getName instead}}

(const void *)sel_registerName("foo");  // ok

(void) s;   // ok

(void *const)s; // ok

(const void *const)s; // ok
}
