// RUN: %clang_cc1 -std=c11 -fsyntax-only -Wno-gcc-compat -verify=expected,implicit %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -Wno-gcc-compat -DDECLARE_CFSTRING -verify %s

// An implicitly added format_arg attribute may refer to a missing argument,
// both during error recovery and with a non-prototype declaration.
// https://github.com/llvm/llvm-project/issues/225034
#ifdef DECLARE_CFSTRING
char *__CFStringMakeConstantString();
#endif

void a(char *) __attribute__((format(__CFString__, 1, 2))); // implicit-note {{passing argument to parameter here}}

void b(void) {
  a(__CFStringMakeConstantString()); // expected-warning {{format string is not a string literal (potentially insecure)}}
  // expected-note@-1 {{treat the string as an argument to avoid this}}
  // implicit-error@-2 {{call to undeclared function '__CFStringMakeConstantString'}}
  // implicit-error@-3 {{incompatible integer to pointer conversion passing 'int' to parameter of type 'char *'}}
}
