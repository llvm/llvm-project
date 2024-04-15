// This test verifies that config macro warnings are emitted when it looks like
// the user expected a `#define` to impact the import of a module.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Prebuild the `config` module so it's in the module cache.
// RUN: %clang_cc1 -std=c99 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -DWANT_FOO=1 -emit-module -fmodule-name=config %t/module.modulemap

// Verify that each time the `config` module is imported the current macro state
// is checked.
// RUN: %clang_cc1 -std=c99 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %t -DWANT_FOO=1 %t/config.m -verify

// Verify that warnings are emitted before building a module in case the command
// line macro state causes the module to fail to build.
// RUN: %clang_cc1 -std=c99 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %t %t/config_error.m -verify

//--- module.modulemap

module config {
  header "config.h"
  config_macros [exhaustive] WANT_FOO, WANT_BAR
}

module config_error {
  header "config_error.h"
  config_macros SOME_VALUE
}

//--- config.h

#ifdef WANT_FOO
int* foo(void);
#endif

#ifdef WANT_BAR
char *bar(void);
#endif

//--- config_error.h

struct my_thing {
  char buf[SOME_VALUE];
};

//--- config.m

@import config;

int *test_foo(void) {
  return foo();
}

char *test_bar(void) {
  return bar(); // expected-error{{call to undeclared function 'bar'; ISO C99 and later do not support implicit function declarations}} \
                // expected-error{{incompatible integer to pointer conversion}}
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

//--- config_error.m

#define SOME_VALUE 5 // expected-note{{macro was defined here}}
@import config_error; // expected-error{{could not build module}} \
                      // expected-warning{{definition of configuration macro 'SOME_VALUE' has no effect on the import of 'config_error';}}
