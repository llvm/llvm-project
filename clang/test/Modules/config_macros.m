@import config;

int *test_foo(void) {
  return foo();
}

char *test_bar(void) {
  return bar(); // expected-warning{{implicit declaration of function 'bar' is invalid in C99}} \
                // expected-warning{{incompatible integer to pointer conversion}}
}

#undef WANT_FOO // expected-note{{macro was #undef'd here}}
@import config; // expected-warning{{#undef of configuration macro 'WANT_FOO' has no effect on the import of 'config'; pass '-UWANT_FOO' on the command line to configure the module}}

#define WANT_FOO 2 // expected-note{{macro was defined here}}
@import config; // expected-warning{{definition of configuration macro 'WANT_FOO' has no effect on the import of 'config'; pass '-DWANT_FOO=...' on the command line to configure the module}}

#undef WANT_FOO
#define WANT_FOO 1
@import config; // okay

#define WANT_BAR 1 // expected-note{{macro was defined here}}
@import config; // expected-warning{{definition of configuration macro 'WANT_BAR' has no effect on the import of 'config'; pass '-DWANT_BAR=...' on the command line to configure the module}}

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -DWANT_FOO=1 -emit-module -fmodule-name=config %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -DWANT_FOO=1 %s -verify

