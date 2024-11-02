// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=diag_pragma %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -verify -fmodules-cache-path=%t -I %S/Inputs %s
// FIXME: When we have a syntax for modules in C, use that.

@import diag_pragma;

int foo(int x) {
  if (x = DIAG_PRAGMA_MACRO) // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                             // expected-note {{place parentheses}} expected-note {{use '=='}}
    return 0;
  return 1;
}
