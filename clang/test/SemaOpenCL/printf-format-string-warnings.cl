// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0 -finclude-default-header

// Make sure warnings are produced based on printf format strings.

kernel void format_string_warnings(__constant char* arg) {

  printf("%d", arg); // expected-warning {{format specifies type 'int' but the argument has type '__constant char *'}}

  printf("not enough arguments %d %d", 4); // expected-warning {{more '%' conversions than data arguments}}

  printf("too many arguments", 4); // expected-warning {{data argument not used by format string}}
}
