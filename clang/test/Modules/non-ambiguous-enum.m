// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F%S/Inputs/non-ambiguous-enum -fsyntax-only %s -verify
#import <B/b.h>
#import <A/a.h>

// expected-no-diagnostics

int foo() {
  return MyEnumCst;
}
