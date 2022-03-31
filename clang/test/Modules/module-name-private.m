// RUN: rm -rf %t
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-canonical -fsyntax-only %s -Wprivate-module -fmodule-name=A -Rmodule-build

// Because of -fmodule-name=A, no module (A or A_Private) is supposed to be
// built and -Rmodule-build should not produce any output.

// expected-no-diagnostics

#import <A/a.h>
#import <A/aprivate.h>

int foo(void) { return APRIVATE; }
